from util import *
from model2 import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
import torchaudio
import torchaudio.transforms as AT
import torch
import pdb
import logging
import argparse
import time
import random
import json

ID2CLS = {0:'correct', 1:'incorrect', 2:'prompt', 3:'repeat', 4:'skip', 5:'stutter', 6:'tracking'}
ID2DET = {0:'fluent', 1:'not fluent'}

def clean_str(text):
    text = re.sub('[^A-Za-z0-9\s]+','',text)
    return text.lower().strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-path', type=str, default='',
                        help='path to recorded audio')
    parser.add_argument('--text', type=str, default='',
                        help='passage that was read')
    parser.add_argument('--tokenizer-path', type=str, default='tokenizers/librispeech.json',
                        help='path to pretrained wordpiece tokenizer')
    args = parser.parse_args()

    ## Load pretrained models
    print(f'Loading pretrained models')
    device = torch.device("cpu")
    tokenizer_path = "tokenizers/librispeech.json"
    device = torch.device("cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = 2+tokenizer.get_vocab_size()
    config = {'embed_dim':320, 'vocab_size':vocab_size, 'dropout':0.1, 'input_dim':80, 'pyr_layer':3, 'nlayer':6, 'multi-gpu':False, 'nhead':1, 'nclasses':7, 'pretrain':False}
    model_path = "/data/data24/scratch/sunderv/saved_models/deploy_readr.pt_best.pt"
    norm_path = "/data/data24/scratch/sunderv/saved_models/readr.pickle"
    model = Detector(config)
    model.eval()
    load_dict(model, model_path, loc=f'cpu')
    norm = load_pick(norm_path)

    ## Load data
    print(f'Loading data')
    audio_path = args.audio_path
    text = args.text
    compute_stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
    compute_fbanks = Filterbank(n_mels=80)
    wav, org_sr = torchaudio.load(audio_path)
    wav = AT.Resample(org_sr, 16000)(wav)
    features = compute_stft(wav)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)
    text = clean_str(text)

    ## Setup data
    print(f'Setting up data')
    speech_batch = [pad(features.squeeze(0), factor=8).unsqueeze(0)]
    X, lens, lmax = padding(speech_batch)
    lens_s = torch.tensor(lens)
    text_raw = [text]
    tokenized = tokenizer.encode_batch(text_raw)

    text_batch = pack_sequence([torch.tensor([30000]+x.ids).long() for x in tokenized], enforce_sorted=False)
    text_batch, lens_t = pad_packed_sequence(text_batch, batch_first=True)            
    merge_idx = [merge(x.tokens) for x in tokenized]

    ## Pass to model
    print(f'Sending data through model\n')
    lens_norm = lens_s / lmax
    sbatch = norm(X, lens_norm.float(), epoch=1000)
    with torch.no_grad():
        pred_cat, pred_det, _ = model.decouple_hier(sbatch, text_batch, lens_s, lens_t, merge_idx)
    y_det = torch.max(pred_det, dim=1)[1].cpu().tolist()
    y_det = [ID2DET[x] for x in y_det]
    y_cls = []
    for i, detection in enumerate(y_det):
        if detection == 'not fluent':
            y_cls.append(ID2CLS[torch.argmax(pred_cat[i][1:]).item() + 1])
        else:
            y_cls.append(ID2CLS[0])
    words = text.split()
    print(f'Detection results = {list(zip(words,y_det))}')
    print(f'Classification results = {list(zip(words,y_cls))}')

if __name__ == '__main__':
    main()
