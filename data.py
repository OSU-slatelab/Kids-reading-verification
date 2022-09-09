from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from itertools import chain
from tokenizers import Tokenizer
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
from speechbrain.lobes.augment import SpecAugment
from speechbrain.lobes.features import MFCC
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

MAPPING = {'correct':0, 'incorrect':1, 'prompt':2, 'repeat':3, 'skip':4, 'stutter':5, 'tracking':6}

def crop(signal, length):
    length_adj = signal.shape[1] - length
    if length_adj > 0:
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        return signal[:,start:start + length]
    return signal

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

def simplify_labels2(lbl):
    if lbl != 'correct':
        return 1
    else:
        return 0

def map_labels(lst):
    lst2 = []
    for l in lst:
        lst2.append(MAPPING[l])
    return lst2

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=16000, train=False, avg_dur=200.0):
        self.max_len = int(sample_rate*avg_dur)
        self.df = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate
        self.args = args
        self.train = train

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        #rest = row['audio_file'][39:]
        #rest_ = rest.split('/')
        #rest = rest_[0]+'/'+rest_[1]+'/'+rest_[-1]
        #audio_file_path = os.path.join('/data/data26/scratch/lvenkat/librispeech/LibriKids',rest)
        #try:
        wav, org_sr = torchaudio.load(row['audio_file'])
        #if self.train:
        #    wav = crop(wav, self.max_len)
        #except:
        #    return None, None, None
        #if org_sr > self.sr:
        wav = AT.Resample(org_sr, self.sr)(wav)
        #if self.train:
        #    wav = crop(wav, self.max_len)
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance']), row['label']
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), map_labels(ast.literal_eval(row['label'])), simplify_labels(ast.literal_eval(row['label'])) #map_labels(ast.literal_eval(row['label']))#, simplify_labels(ast.literal_eval(row['label']))#map_labels(ast.literal_eval(row['label'])), simplify_labels(ast.literal_eval(row['label']))

class WordDataset(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=16000, train=False, avg_dur=200.0):
        super(WordDataset, self).__init__(args, csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['main_wav'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
            org_sr = self.sr
        crop_start_ix = int(row['begin']*org_sr)
        crop_end_ix = int(row['end']*org_sr)
        wav = wav[:,crop_start_ix:crop_end_ix]
        return self.get_filterbanks(wav), simplify_labels2(row['label'])#MAPPING[row['label']]#simplify_labels2(row['label'])#MAPPING[row['label']]

class PhonetDataset(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=16000, train=False, avg_dur=200.0):
        super(PhonetDataset, self).__init__(args, csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(os.path.join('/data/data25/scratch/sunderv/rraces/forced_aligned/wav_cmu/', row['wav_file']))
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
            org_sr = self.sr
        feature_maker = MFCC(deltas=False, context=False, sample_rate=org_sr)
        feats = feature_maker(wav)
        ph_post = torch.from_numpy(np.load(os.path.join('/data/data25/scratch/sunderv/rraces/forced_aligned/posterior_cmu/', row['post_path']))).unsqueeze(0)
        phones = ast.literal_eval(row['phones'])
        return feats, ph_post, phones, simplify_labels2(row['label'])#simplify_labels2(row['label'])#MAPPING[row['label']]
        
class WordCollator(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0] is not None and x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)
        cls_target = torch.cat([torch.tensor([x[1]]) for x in lst if x[0].size(1) > 2])
        return X, torch.tensor(lens), lmax, cls_target

class PhonetCollator(object):
    def __init__(self, args):
        self.args = args
        self.p2i = json.loads(open(args.vocab_path).readline())

    def __call__(self, lst):
        speech_batch = [x[0] for x in lst if x[0].size(1) > 2]
        X, lens_mfcc, lmax = padding(speech_batch)
        post_batch = [x[1] for x in lst if x[0].size(1) > 2]
        X_post, lens_post, _ = padding(post_batch)
        pack_phone = pack_sequence([torch.tensor([self.p2i[k] for k in x[2]]).long() for x in lst if x[0].size(1) > 2], enforce_sorted=False)
        phone_batch, lens_ph = pad_packed_sequence(pack_phone, batch_first=True)
        cls_target = torch.cat([torch.tensor([x[3]]) for x in lst if x[0].size(1) > 2])
        return X, X_post, phone_batch, torch.tensor(lens_mfcc), lmax, torch.tensor(lens_post), lens_ph, cls_target

class Collator(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenizer.from_file(args.tokenizer_path)

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)
        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0] is not None and x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)
        text_raw = [x[1] for x in lst if x[0] is not None and x[0].size(1) > 2]
        tokenized = self.tokenizer.encode_batch(text_raw)
        if not self.args.pretrain:
            text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
            text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
            token_target = [torch.tensor(x.ids+[30001]).long() for x in tokenized]
            token_target = torch.cat(token_target)
            merge_idx = [merge(x.tokens) for x in tokenized]
            cls_target = torch.cat([torch.tensor(x[2]) for x in lst if x[0].size(1) > 2])
            for i, x in enumerate(lst):
                if len(merge_idx[i]) != len(x[2]):
                    print(merge_idx[i],'---',x[2],'---',text_raw[i]) 
        else:
            text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
            text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
            token_target = [torch.tensor(x.ids+[30001]).long() for x in tokenized]
            token_target = torch.cat(token_target)
            merge_idx = None
            cls_target = None

        return X, torch.tensor(lens), lmax, text_batch, lens_t, token_target, merge_idx, cls_target

class CollatorDec(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenizer.from_file(args.tokenizer_path)

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)
        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0] is not None and x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)
        text_raw = [x[1] for x in lst if x[0] is not None and x[0].size(1) > 2]
        tokenized = self.tokenizer.encode_batch(text_raw)

        text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
        text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
        token_target = [torch.tensor(x.ids+[30001]).long() for x in tokenized]
        token_target = torch.cat(token_target)
        merge_idx = [merge(x.tokens) for x in tokenized]
        cat_target = torch.cat([torch.tensor(x[2]) for x in lst if x[0].size(1) > 2])
        det_target = torch.cat([torch.tensor(x[3]) for x in lst if x[0].size(1) > 2])

        return X, torch.tensor(lens), lmax, text_batch, lens_t, token_target, merge_idx, cat_target, det_target

