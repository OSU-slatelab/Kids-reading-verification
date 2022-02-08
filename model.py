from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
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

class RNN(nn.Module):
    def __init__(self, inp_size, nhid, nlayer=1, dropout=0.1):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(inp_size, nhid, nlayer, bidirectional=False, dropout=dropout)
        self.nlayer = nlayer
        self.nhid = nhid

    def forward(self, input):
        pack = pack_sequence(input, enforce_sorted=False)
        output, (hn, cn) = self.rnn(pack, (h0, c0))
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        #hn = hn.view(self.nlayer, 2, len(input), self.nhid)
        #hn_l = torch.cat([hn[-1][0], hn[-1][1]], dim=1)
        return seq_unpacked, lens_unpacked #hn[-1,0,:,:] + hn[-1,1,:,:]

class pLSTMLayerParallel(nn.Module):
    def __init__(self, input_feature_dim, dropout_rate=0.1):
        super(pLSTMLayerParallel, self).__init__()
        self.pLSTM = nn.LSTM(input_feature_dim, input_feature_dim, 1, bidirectional=False, dropout=dropout_rate, batch_first=False)
        self.norm = nn.LayerNorm(input_feature_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_padded, lens):
        lens = lens.cpu()
        pack = pack_padded_sequence(input_padded, lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.pLSTM(pack)
        output_padded, lens_unpacked = pad_packed_sequence(output, batch_first=True)
        # residual connection b/w subsequent layers
        output_padded = self.norm(output_padded+self.dropout(input_padded))
        output_padded_batchf = self.dropout(output_padded)
        return output_padded_batchf


# LSTM layer for pLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through pLSTM
# Note the input should have timestep%2 == 0
class pLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, rnn_unit='LSTM', dropout_rate=0.0, add_norm=True):
        super(pLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        #self.pLSTM = self.rnn_unit(input_feature_dim, input_feature_dim, 1, bidirectional=False, dropout=dropout_rate, batch_first=False)
        #self.norm = nn.LayerNorm(input_feature_dim)
        #self.dropout = nn.Dropout(dropout_rate)
        #self.add_norm = add_norm
        self.pLSTM = nn.DataParallel(pLSTMLayerParallel(input_feature_dim), device_ids=[0,1])
    
    def forward(self, input_x):
        batch_size = len(input_x)
        red_x = []
        # Reduce time resolution
        for x in input_x:
            timestep = x.size(0)
            feature_dim = x.size(1)
            red_x.append(x.contiguous().view(int(timestep/2), feature_dim*2))
        pack = pack_sequence(red_x, enforce_sorted=False)
        padded, lens = pad_packed_sequence(pack, batch_first=True)
        output_padded_batchf = self.pLSTM(padded, lens)
        out_list = [output_padded_batchf[i][:l] for i, l in enumerate(lens.long().cpu().tolist())]
        return out_list, hidden
        
        #pack = pack_sequence(red_x, enforce_sorted=False)
        #output, hidden = self.pLSTM(pack)
        #output_padded, lens_unpacked = pad_packed_sequence(output)
        ## residual connection b/w subsequent layers
        #if self.add_norm:
        #    input_padded, _ = pad_packed_sequence(pack) 
        #    output_padded = self.norm(output_padded+self.dropout(input_padded)) 
        #output_padded_batchf = self.dropout(output_padded).permute(1,0,2)
        #out_list = [output_padded_batchf[i][:l] for i, l in enumerate(list(lens_unpacked))]
        #return out_list, hidden

# Listener is a pLSTM stacking n layers to reduce time resolution 2^n times
class Listener(nn.Module):
    def __init__(self, input_dim, listener_layer, rnn_unit, device=None, dropout_rate=0.1):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        listener_hidden_dim = 2*input_dim
        self.pLSTM_layer0 = pLSTMLayer(listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        dim = listener_hidden_dim
        for i in range(1,self.listener_layer):
            dim = 2*dim
            setattr(self, 'pLSTM_layer'+str(i), pLSTMLayer(dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    def forward(self, input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.listener_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)

        packed = pack_sequence(output, enforce_sorted=False)
        output_padded, lens = pad_packed_sequence(packed)
        mask = get_mask(list(lens))
        return output_padded, mask.to(self.device)

class Attention(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=1280, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn

class Reader(nn.Module):
    def __init__(self, embed_dim, vocab_size, device, dropout_rate=0.1):
        super(Reader, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, 2*embed_dim, 1, bidirectional=False, dropout=dropout_rate, batch_first=False)
        self.norm = nn.LayerNorm(2*embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    def forward(self, input_x):
        embed_in = [self.embedding(x) for x in input_x]
        pack = pack_sequence(embed_in, enforce_sorted=False)
        out_pack, _ = self.encoder(pack)
        output, lens = pad_packed_sequence(out_pack)
        output = self.norm(output)
        mask = get_mask(list(lens))
        return output, mask.to(self.device)

class ASR(nn.Module):
    def __init__(self, config):
        super(ASR, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['device'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['listener_layer'], 'LSTM', config['device'], config['dropout'])
        attention_indim = (2**config['listener_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, dim_feedforward=2*attention_indim, dropout_rate=config['dropout'])
        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False, dropout=config['dropout'])
        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, input_t, merge_idx=None):
        listened, mask_s = self.listener(input_s)
        read, mask_t = self.reader(input_t)

        aligned_tq, attn_tq = self.attention(read, listened, mask_s.bool())

        out_feat, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read], dim=2)))
        out = extract(self.classifier_tok(self.dropout(out_feat)).permute(1,0,2), mask_t.long())

        return out, None, attn_tq

class Detector(nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.reader = Reader(config['embed_dim'], config['vocab_size'], config['device'], config['dropout'])
        self.listener = Listener(config['input_dim'], config['listener_layer'], 'LSTM', config['device'], config['dropout'])
        attention_indim = (2**config['listener_layer'])*config['input_dim']
        self.attention = Attention(attention_indim, dim_feedforward=2*attention_indim, dropout_rate=config['dropout'])
        self.rnn = nn.LSTM(2*attention_indim, attention_indim, 1, bidirectional=False, dropout=config['dropout'])
        self.rnn_cls = nn.LSTM(attention_indim, attention_indim, 1, bidirectional=False, dropout=config['dropout'])
        self.classifier_tok = nn.Linear(attention_indim, config['vocab_size'])
        self.classifier_cls = nn.Linear(attention_indim, config['nclasses'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, input_s, input_t, merge_idx=None):
        listened, mask_s = self.listener(input_s)
        read, mask_t = self.reader(input_t)

        aligned_tq, attn_tq = self.attention(read, listened, mask_s.bool())

        out_tok_seq, _ = self.rnn(self.dropout(torch.cat([aligned_tq,read], dim=2)))
   
        out_tok = extract(self.classifier_tok(self.dropout(out_tok_seq)).permute(1,0,2), mask_t.long())

        out_feat, _ = self.rnn_cls(self.dropout(out_tok_seq[1:,:,:]))
        out_cls = self.classifier_cls(merge(out_feat.permute(1,0,2), merge_idx))
        return out_cls, out_tok, attn_tq

