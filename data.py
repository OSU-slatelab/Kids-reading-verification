from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from itertools import chain
from tokenizers import Tokenizer
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
from speechbrain.lobes.augment import SpecAugment
import torch
import random
import json
import ast
import os
import time
import pandas as pd
import pdb
import numpy as np
import torchaudio
import torchaudio.transforms as AT
import copy
import re

def pad(input, factor=4):
    add_size = input.size(0) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(rem_size, input.size(1))], dim=0)
    else:
        return input

def list_batch(X, lens, lmax):
    idx = list(range(len(lens)))
    random.shuffle(idx)
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i,:l,:])
    return sbatch_

def padding(sbatch):
    dim = sbatch[0].size(2)
    lens = [x.size(1) for x in sbatch]
    lmax = max(lens)
    padded = []
    for x in sbatch:
        pad = torch.zeros(lmax, dim)
        pad[:x.size(1),:] = x
        padded.append(pad.unsqueeze(0))
    X = torch.cat(padded, dim=0)
    return X, lens, lmax

def clean_str(text):
    text = re.sub('[^A-Za-z0-9\s]+','',text)
    return text.lower().strip()

def merge(lst):
    def merge_sub(i, lst):
        x = [i] 
        while i+1 < len(lst) and lst[i+1][0] == '#':
            x.append(i+1)
            i = i+1
        return x,i+1
    x, j = merge_sub(0, lst)
    idx = [x]
    while j < len(lst):
        x, j = merge_sub(j, lst)
        idx.append(x)
    return idx

def simplify_labels(lst):
    lst2 = []
    for l in lst:
        if l != 'correct':
            lst2.append(1)
        else:
            lst2.append(0)
    return lst2

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate
        self.args = args

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['audio_file'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance']), row['label']
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), simplify_labels(ast.literal_eval(row['label']))

class Collator(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenizer.from_file(args.tokenizer_path)

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)
        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)
        text_raw = [x[1] for x in lst if x[0].size(1) > 2]
        tokenized = self.tokenizer.encode_batch(text_raw)
        if not self.args.pretrain:
            text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
            text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
            token_target = [torch.tensor(x.ids+[30001]).long() for x in tokenized]
            token_target = torch.cat(token_target)
            merge_idx = [merge(x.tokens) for x in tokenized]
            cls_target = torch.cat([torch.tensor(x[2]) for x in lst if x[0].size(1) > 2])
        else:
            text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
            text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
            token_target = [torch.tensor(x.ids+[30001]).long() for x in tokenized]
            token_target = torch.cat(token_target)
            merge_idx = None
            cls_target = None

        return X, torch.tensor(lens), lmax, text_batch, lens_t, token_target, merge_idx, cls_target
