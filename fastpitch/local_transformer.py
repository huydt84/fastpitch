import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
import random

from fastpitch.local_attention.local_attention import LocalAttention
from fastpitch.transformer import PositionalEmbedding, PositionwiseConvFF

from common.utils import mask_from_lens

# helper function

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling functions

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# multi-head attention

class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.1,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = default(exact_windowsize, True),
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, attn_bias = None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# dynamic positional bias

class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads)
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, i, j):
        device = self.device
        assert j >= i

        rel_dist = torch.arange(j, dtype = torch.float, device = device)
        bias = self.mlp(rearrange(rel_dist, '... -> ... 1'))

        i_seq = torch.arange(j - i, j, device = device)
        j_seq = torch.arange(j, device = device)

        rel_dist_indices = (rearrange(i_seq, 'i -> i 1') - rearrange(j_seq, 'j -> 1 j')).abs()

        bias = rearrange(bias[rel_dist_indices], 'i j h -> h i j')
        return bias

class LocalTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        attn_dropout,
        ff_droupout,
        causal,
        window_size,
        use_xpos,
        xpos_scale_base,
        use_rotary_pos_emb,
        mult,
        prenorm=True,
        **kwargs
    ):
        super().__init__()
        self.attn = LocalMHA(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal, window_size = window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base, use_rotary_pos_emb = use_rotary_pos_emb, prenorm = prenorm, **kwargs)
        self.ff = FeedForward(dim = dim, mult = mult, dropout = ff_droupout)
        
    def forward(self, x, mask, attn_bias):
        x = self.attn(x, mask=mask, attn_bias = attn_bias) + x
        x = self.ff(x) + x
        
        return x
        

# main transformer class

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        causal = False,
        local_attn_window_size = 16,
        dim_head = 64,
        heads = 1,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ignore_index = -1,
        use_xpos = False,
        xpos_scale_base = None,
        use_dynamic_pos_bias = False,
        group=3,
        padding_idx=0,
        embed_input=True,
        **kwargs
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.group = group
        self.n_layer = depth
        if embed_input:
            self.word_emb = nn.Embedding(num_tokens, dim,
                                         padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(dim)

        self.layers = nn.ModuleList()

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(
                LocalTransformerLayer(dim=dim, dim_head=dim_head, heads=heads, 
                                      attn_dropout=attn_dropout, 
                                      ff_droupout=ff_dropout, causal=causal,
                                      window_size=local_attn_window_size, 
                                      use_xpos=use_xpos, xpos_scale_base=xpos_scale_base,
                                      use_rotary_pos_emb=not use_dynamic_pos_bias,
                                      mult=ff_mult, prenorm=True)
            )

        self.ignore_index = ignore_index
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(self, x, seq_lens=None, mask=None, conditioning=0):
        n, device = x.shape[1], x.device
        
        if self.word_emb is not None:
            mask = (x != self.padding_idx).unsqueeze(2)
            x = self.word_emb(x)           
        else:
            mask = mask_from_lens(seq_lens).unsqueeze(2)

        x = x + self.pos_emb(torch.arange(n, device = device).to(x.dtype)) + conditioning

        # dynamic pos bias

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers
        temp_list = [i for i in range(self.n_layer)]
        layer_list = []
        
        if self.training and self.group != -1:
            for i in range(self.n_layer // self.group):
                temp = temp_list[i * self.group:(i + 1) * self.group]
                random.shuffle(temp)
                layer_list.extend(temp)
        else:
            layer_list = [i for i in range(self.n_layer)]
        
        for layer in layer_list:
            x = self.layers[layer](x, mask, attn_bias = attn_bias)

        logits = self.to_logits(x)

        return logits, mask