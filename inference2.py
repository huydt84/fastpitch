# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm
import json
import re

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.functional import l1_loss
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

import models
from common import gpu_affinity
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
from fastpitch.pitch_transform import pitch_transform_custom
from hifigan.data_function import MAX_WAV_VALUE, mel_spectrogram
from hifigan.models import Denoiser
from waveglow import model as glow

def prepare_input_sequence(text_list, device, symbol_set, text_cleaners, batch_size=16, p_arpabet=0.0):
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)
    
    input = [torch.LongTensor(tp.encode_text(text)) for text in text_list]
    
    for t in input:
        print(tp.sequence_to_text(t.numpy()))
         
    batches = []
    for b in range(0, len(input), batch_size):
        batch = input[b:b+batch_size]        
        batch = pad_sequence(batch, batch_first=True)
        
        if type(batch) is torch.Tensor:
            batch = batch.to(device)
            
        batches.append(batch)
        
    return batches

def load_pretrained_weights(model, ckpt_fpath):
    model = getattr(model, "module", model)
    weights = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
    weights = {re.sub("^module.", "", k): v for k, v in weights.items()}

    ckpt_emb = weights["encoder.word_emb.weight"]
    new_emb = model.state_dict()["encoder.word_emb.weight"]

    ckpt_vocab_size = ckpt_emb.size(0)
    new_vocab_size = new_emb.size(0)
    if ckpt_vocab_size != new_vocab_size:
        print("WARNING: Resuming from a checkpoint with a different size "
            "of embedding table. For best results, extend the vocab "
            "and ensure the common symbols' indices match.")
        min_len = min(ckpt_vocab_size, new_vocab_size)
        weights["encoder.word_emb.weight"] = ckpt_emb if ckpt_vocab_size > new_vocab_size else new_emb
        weights["encoder.word_emb.weight"][:min_len] = ckpt_emb[:min_len]

    model.load_state_dict(weights)

def load_hifigan(checkpoint, model, amp, key="generator"):
    checkpoint_data = torch.load(checkpoint)

    sd = checkpoint_data[key]
    sd = {re.sub('^module\.', '', k): v for k, v in sd.items()}
    status = model.load_state_dict(sd, strict=False)
    return model


class TTS():
    def __init__(self, model_path="FastPitch_checkpoint_teacher_100.pt", model_config="fastpitch_teacher.json", 
                    vocoder_path="hifigan-v2/generator_v2", vocoder_config="hifigan-v2/config.json", 
                    device="cpu", amp=True, denoising_strength=0.0,
                    symbol_set="english_basic", text_cleaners=['english_cleaners_v2'], p_arpabet=0.0):
        model_config = json.load(open(model_config)) 
        self.model = models.get_model('FastPitch', model_config, device=device)
        load_pretrained_weights(self.model, model_path)
        self.model.eval()

        vocoder_config = json.load(open(vocoder_config)) 
        self.vocoder = models.get_model('HiFi-GAN', vocoder_config, device=device)

        if vocoder_path is not None:
            self.vocoder = load_hifigan(vocoder_path, self.vocoder, amp)
            self.vocoder.eval()
        else:
            raise Exception("vocoder_path is required")

        self.denoising_strength = denoising_strength
        self.denoiser = None
        if denoising_strength > 0.0:
            self.denoiser = Denoiser(self.vocoder).to(device)

        self.tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)
        self.device = device

    @torch.no_grad()
    def generate_audio(self, text, output="audio.wav", max_wav_value=32768.0, fade_out=10, hop_length=256, sampling_rate=22050):
        start = time.time()
        for _ in range(10):
            mel, mel_lens, *_ = self.model.infer(self.encode(text))
            audios = self.vocoder(mel).float()
            if self.denoiser is not None:
                audios = self.denoiser(audios.squeeze(1), self.denoising_strength)
            audios = audios.squeeze(1) * max_wav_value
        end = time.time()

        if output is not None:
            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * hop_length]

                if fade_out:
                    fade_len = fade_out * hop_length
                    fade_w = torch.linspace(1.0, 0.0, fade_len)
                    audio[-fade_len:] *= fade_w.to(audio.device)

                audio = audio / torch.max(torch.abs(audio))
                audio_path = Path(output)
                write(audio_path, sampling_rate, audio.cpu().numpy())

        return (end - start) / 10

    def encode(self, text):
        input = [torch.LongTensor(self.tp.encode_text(text))]
        input = pad_sequence(input, batch_first=True)
        input = input.to(self.device)

        return input
