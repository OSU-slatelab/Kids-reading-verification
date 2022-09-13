import torch
import pickle
import pandas as pd
import pdb
from tqdm import tqdm
import torch.nn as nn

class FScore(nn.Module):
    def __init__(self, ncls=7, temp=1.0):
        super(FScore, self).__init__()
        self.temp = torch.tensor([[1.0, 0.1, 0.5, 0.2, 0.07, 0.05, 0.01]])
        self.ncls = 7

    def mhot_tgt(self, tlist):
        tgt = torch.zeros(len(tlist), self.ncls)
        for i,t in enumerate(tlist):
            tgt[i][t] = 1.
        return tgt

    def forward(self, pred, y_true): # pred --> (32, 7) y_true --> (32,)
        pred = torch.softmax(pred / self.temp.to(pred.get_device()), dim=1)
        pred = pred / torch.max(pred, dim=1, keepdim=True)[0]
        y_true_oh = self.mhot_tgt(y_true.cpu().tolist()).to(pred.get_device())
        TP = (pred*y_true_oh).sum(dim=0, keepdim=True)
        TPFP = pred.sum(dim=0, keepdim=True)
        Nk = y_true_oh.sum(dim=0, keepdim=True)
        return 1. - (2*TP/(TPFP + Nk + 1e-6)).mean()

class CBCrossEntropy(nn.Module):
    def __init__(self, ny, device, beta=0, gamma=0):
        super(CBCrossEntropy, self).__init__()  
        ny = torch.tensor(ny).unsqueeze(0).to(device) # (1,nclasses)
        self.W = (1. - beta) / (1. - (beta ** ny))
        self.gamma = gamma

    def forward(self, pred, gt):
        pred = -((1. - torch.softmax(pred, dim=1))**self.gamma) * (torch.log_softmax(pred, dim=1))
        pred = pred * self.W
        loss = torch.gather(pred, 1, gt.unsqueeze(1)).squeeze().mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt):
        pred = -((1. - torch.softmax(pred, dim=1))**self.gamma) * (torch.log_softmax(pred, dim=1))
        loss = torch.gather(pred, 1, gt.unsqueeze(1)).squeeze().mean()
        return loss

def get_params(model):
    low = []
    high = []
    for name, parameter in model.named_parameters():
        if 'text_encoder' in name:
            low.append(parameter)
        else:
            high.append(parameter)
    return low, high

def load_dict(model, dict_path, loc='cuda:0', ddp=False):
    pretrained_dict = torch.load(dict_path, map_location=loc)
    model_dict = model.state_dict()
    new_pt_dict = {}
    for k, v in pretrained_dict.items():
        k_new = k
        if not ddp and k[:7] == 'module.':
            k_new = k[7:]
        elif ddp and k[:7] != 'module.':
            k_new = f'module.{k}'
        new_pt_dict[k_new] = v
    pretrained_dict = {k: v for k, v in new_pt_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def save_pick(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pick(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def save_checkpoint(state, filename):
    torch.save(state, filename)
