from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from lstm_util import *
import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F


def unpack_speech(input_s, layer=3):
    for i in range(layer):
        timestep = input_s.size(0)
        feature_dim = input_s.size(1)
        input_s = input_s.contiguous().view(int(timestep/2), feature_dim*2)
    return input_s

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def mean_pool(tens, mask):
    return (tens*(1.-mask).t().unsqueeze(2)).sum(dim=0) / (1.-mask).sum(dim=1,keepdim=True)

def extract(tens, mask):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]])
    return torch.cat(out, dim=0)

def merge(tens, merge_idx):
    out = []
    for bat, idx_bat in enumerate(merge_idx):
        for seq, idx_seq in enumerate(idx_bat):
            out.append(tens[bat][idx_seq].mean(dim=0, keepdim=True))
    return torch.cat(out, dim=0)

def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

class Listener(nn.Module):
    def __init__(self, input_dim, pyr_layer, nlayer, dropout=0.1):
        super(Listener, self).__init__()
        self.pyr_layer = pyr_layer
        self.p_encoder = pLSTM(input_dim, pyr_layer, dropout=dropout)
        self.encoder = CustomLSTM((2**pyr_layer)*input_dim, nlayer, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens):
        lens_org = (copy.deepcopy(lens) / (2**self.pyr_layer)).long()
        out_pyr, lens = self.p_encoder(input_x, lens)
        out_lstm, _ = self.encoder(out_pyr, lens)
        return out_lstm, lens_org

class Attention(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=1280, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn

class Reader(nn.Module):
    def __init__(self, embed_dim, vocab_size, dropout=0.1):
        super(Reader, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, 2*embed_dim, 1, bidirectional=False)
        self.norm = nn.LayerNorm(2*embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens): # bsz, seq_len
        lens_org = (copy.deepcopy(lens)).long()
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        embed_in = self.embedding(input_x)
        # pack sequence
        pack = pack_padded_sequence(embed_in, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.encoder.flatten_parameters()
        output, hidden = self.encoder(pack)
        # pad packed seq output of LSTM
        out_pad, lens = pad_packed_sequence(output, batch_first=True)       
        output = self.norm(out_pad)
        return output, lens_org

class ASR(nn.Module):
    def __init__(self, config):
        super(ASR, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['nlayer'], config['dropout'])

        if config['multi-gpu']:
            self.reader = nn.DataParallel(self.reader, device_ids=[0,1])
            self.listener = nn.DataParallel(self.listener, device_ids=[0,1])

        attention_indim = (2**config['pyr_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, nhead=config['nhead'], dim_feedforward=2*attention_indim, dropout=config['dropout'])

        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False)
        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, input_t, lens_s, lens_t, merge_idx=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(read.get_device()), get_mask(lens_t_.cpu().tolist()).to(read.get_device())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_feat, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out = extract(self.classifier_tok(self.dropout(out_feat)).permute(1,0,2), mask_t.long())

        return out, None, attn_tq

class Detector(nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['nlayer'], config['dropout'])

        if config['multi-gpu']:
            self.reader = nn.DataParallel(self.reader, device_ids=[0,1])
            self.listener = nn.DataParallel(self.listener, device_ids=[0,1])

        attention_indim = (2**config['pyr_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, nhead=config['nhead'], dim_feedforward=2*attention_indim, dropout=config['dropout'])

        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False)
        self.rnn_cls = nn.LSTM(attention_indim, attention_indim, 1, bidirectional=False)

        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.classifier_cls = nn.Linear(attention_indim, config['nclasses'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, input_t, lens_s, lens_t, merge_idx=None):
        listened, lens_s_ = self.listener(input_s, lens_s)
        read, lens_t_ = self.reader(input_t, lens_t)
        mask_s, mask_t = get_mask(lens_s_.cpu().tolist()).to(read.get_device()), get_mask(lens_t_.cpu().tolist()).to(read.get_device())

        aligned_tq, attn_tq = self.attention(read.permute(1,0,2), listened.permute(1,0,2), mask_s.bool())

        self.rnn.flatten_parameters()
        out_tok_seq, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read.permute(1,0,2)], dim=2)))
        out_tok = extract(self.classifier_tok(self.dropout(out_tok_seq)).permute(1,0,2), mask_t.long())

        self.rnn_cls.flatten_parameters()
        out_feat, _ = self.rnn_cls(self.dropout(out_tok_seq[1:,:,:]))
        out_cls = self.classifier_cls(merge(out_feat.permute(1,0,2), merge_idx))
        return out_cls, out_tok, attn_tq
