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


CHECKPOINT_SPECIFIC_ARGS = [
    'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
    'symbol_set', 'max_wav_value', 'prepend_space_to_text',
    'append_space_to_text']


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, action='append', required=True,
                        help='input text')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true',
                        help='Save generator outputs to disk')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--l2-promote', action='store_true',
                        help='Increase max fetch granularity of GPU L2 cache')
    parser.add_argument('--fastpitch', type=str, default=None, required=False,
                        help='Full path to the spectrogram generator .pt file '
                             '(skip to synthesize from ground truth mels)')
    parser.add_argument('--waveglow', type=str, default=None, required=False,
                        help='Full path to a WaveGlow model .pt file')
    parser.add_argument('-s', '--waveglow-sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('--hifigan', type=str, default=None, required=False,
                        help='Full path to a HiFi-GAN model .pt file')
    parser.add_argument('-d', '--denoising-strength', default=0.0, type=float,
                        help='Capture and subtract model bias to enhance audio')
    parser.add_argument('--hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--win-length', type=int, default=1024,
                        help='STFT win length for denoiser and mel loss')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        choices=[22050, 44100], help='Sampling rate')
    parser.add_argument('--max_wav_value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Run inference with TorchScript model (convert to TS if needed)')
    parser.add_argument('--checkpoint-format', type=str,
                        choices=['pyt', 'ts'], default='pyt',
                        help='Input checkpoint format (PyT or TorchScript)')
    parser.add_argument('--torch-tensorrt', action='store_true',
                        help='Run inference with Torch-TensorRT model (compile beforehand)')
    parser.add_argument('--report-mel-loss', action='store_true',
                        help='Report mel loss in metrics')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--affinity', type=str, default='single',
                        choices=['socket', 'single', 'single_unique',
                                 'socket_unique_interleaved',
                                 'socket_unique_continuous',
                                 'disabled'],
                        help='type of CPU affinity')

    transf = parser.add_argument_group('transform')
    transf.add_argument('--fade-out', type=int, default=10,
                        help='Number of fadeout frames at the end')
    transf.add_argument('--pace', type=float, default=1.0,
                        help='Adjust the pace of speech')
    transf.add_argument('--pitch-transform-flatten', action='store_true',
                        help='Flatten the pitch')
    transf.add_argument('--pitch-transform-invert', action='store_true',
                        help='Invert the pitch wrt mean value')
    transf.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                        help='Multiplicative amplification of pitch variability. '
                             'Typical values are in the range (1.0, 3.0).')
    transf.add_argument('--pitch-transform-shift', type=float, default=0.0,
                        help='Raise/lower the pitch by <hz>')
    transf.add_argument('--pitch-transform-custom', action='store_true',
                        help='Apply the transform from pitch_transform.py')

    txt = parser.add_argument_group('Text processing parameters')
    txt.add_argument('--text-cleaners', type=str, nargs='*',
                     default=['english_cleaners_v2'],
                     help='Type of text cleaners for input text')
    txt.add_argument('--symbol-set', type=str, default='english_basic',
                     help='Define symbol set for input text')
    txt.add_argument('--p-arpabet', type=float, default=0.0, help='')
    txt.add_argument('--heteronyms-path', type=str,
                     default='cmudict/heteronyms', help='')
    txt.add_argument('--cmudict-path', type=str,
                     default='cmudict/cmudict-0.7b', help='')
    return parser

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

def main():
    """
    Launches text-to-speech inference on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    
    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        # print(nproc_per_node)
        affinity = gpu_affinity.set_affinity(
            0,
            nproc_per_node,
            args.affinity
        )
        print(f'Thread affinity: {affinity}')

    if args.l2_promote:
        l2_promote()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.output is not None:
        Path(args.output).mkdir(parents=False, exist_ok=True)
        
    device = torch.device('cuda' if args.cuda else 'cpu')

    gen_train_setup = {}
    voc_train_setup = {}
    generator = None
    vocoder = None
    denoiser = None
    
    is_ts_based_infer = args.torch_tensorrt or args.torchscript
    
    assert args.checkpoint_format == 'pyt' or is_ts_based_infer, \
        'TorchScript checkpoint can be used only for TS or Torch-TRT' \
        ' inference. Please set --torchscript or --torch-tensorrt flag.'

    assert args.waveglow is None or args.hifigan is None, \
        "Specify a single vocoder model"

    def _load_pyt_or_ts_model(model_name, ckpt_path):
        if args.checkpoint_format == 'ts':
            model = models.load_and_setup_ts_model(model_name, ckpt_path,
                                                   args.amp, device)
            model_train_setup = {}
            return model, model_train_setup
        model, _, model_train_setup = models.load_and_setup_model(
            model_name, parser, ckpt_path, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, jitable=is_ts_based_infer)

        if is_ts_based_infer:
            model = torch.jit.script(model)
        return model, model_train_setup

    if args.fastpitch is not None:
        gen_name = 'fastpitch'
        generator, gen_train_setup = _load_pyt_or_ts_model('FastPitch',
                                                           args.fastpitch)

    if args.waveglow is not None:
        voc_name = 'waveglow'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vocoder, _, voc_train_setup = models.load_and_setup_model(
                'WaveGlow', parser, args.waveglow, args.amp, device,
                unk_args=unk_args, forward_is_infer=True, jitable=False)

        if args.denoising_strength > 0.0:
            denoiser = Denoiser(vocoder, sigma=0.0,
                                win_length=args.win_length).to(device)

        if args.torchscript:
            vocoder = torch.jit.script(vocoder)

        def generate_audio(mel):
            audios = vocoder(mel, sigma=args.waveglow_sigma_infer)
            if denoiser is not None:
                audios = denoiser(audios.float(), args.denoising_strength).squeeze(1)
            return audios

    elif args.hifigan is not None:
        voc_name = 'hifigan'
        vocoder, voc_train_setup = _load_pyt_or_ts_model('HiFi-GAN',
                                                         args.hifigan)

        if args.denoising_strength > 0.0:
            denoiser = Denoiser(vocoder, win_length=args.win_length).to(device)

        if args.torch_tensorrt:
            vocoder = models.convert_ts_to_trt('HiFi-GAN', vocoder, parser,
                                               args.amp, unk_args)

        def generate_audio(mel):
            audios = vocoder(mel).float()
            if denoiser is not None:
                audios = denoiser(audios.squeeze(1), args.denoising_strength)
            return audios.squeeze(1) * args.max_wav_value
        
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    for k in CHECKPOINT_SPECIFIC_ARGS:

        v1 = gen_train_setup.get(k, None)
        v2 = voc_train_setup.get(k, None)

        assert v1 is None or v2 is None or v1 == v2, \
            f'{k} mismatch in spectrogram generator and vocoder'

        val = v1 or v2
        if val and getattr(args, k) != val:
            src = 'generator' if v2 is None else 'vocoder'
            print(f'Overwriting args.{k}={getattr(args, k)} with {val} '
                  f'from {src} checkpoint.')
            setattr(args, k, val)
            
    gen_kw = {'pace': args.pace,
              'speaker': args.speaker}
    
    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, args.heteronyms_path)
        
    batches = prepare_input_sequence(args.input, device, args.symbol_set, args.text_cleaners, args.batch_size, args.p_arpabet)
    

