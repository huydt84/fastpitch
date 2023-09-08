# import torch
# from fastpitch.local_attention.local_attention import LocalAttention
# from fastpitch.local_transformer import LocalTransformer
# from fastpitch.transformer import MultiHeadAttn, FFTransformer

# x = torch.LongTensor(torch.randint(0, 148, (10, 855, 1)))

# model_local = LocalTransformer(
#     dim = 384,
#     num_tokens = 148, 
#     depth = 6
# )

# out, mask = model_local(x)
# print(out.shape)

from inference2 import TTS

tts = TTS()

t = tts.generate_audio("She sells seashells by the seashore, shells she sells are great")
print(t)
