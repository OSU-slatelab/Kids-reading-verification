import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from data import *
from model2 import *
import pdb
import argparse

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

class kNN(object):
    def __init__(self, collator, loader_keys, device, norm, bsz=32, ncls=16):
        self.norm = norm
        self.loader_keys = loader_keys
        self.device = device
        self.oh = torch.eye(ncls)

    def get_keys(self, model, loader):
        model.eval()
        keys = []
        i2c = []
        for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(loader):
            i2c.extend(cls_tgt.long().tolist())
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
            label_cls = load2gpu(cls_tgt, self.device)
            label_tok = load2gpu(text_tgt, self.device)
            with torch.no_grad():
                _, _, r, _, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)
            keys.append(r)
        return torch.cat(keys, dim=0), i2c

    def classify(self, model, loader):
        print(f'Encoding memory bank.')
        keys, keys_cls = self.get_keys(model, self.loader_keys)
        keys_cls = torch.tensor(keys_cls)
        print(f'Encoding dev set')
        qrs, qrs_cls = self.get_keys(model, loader)
        print(f'Done.')
        keys = F.normalize(keys, dim=1)
        qrs = F.normalize(qrs, dim=1)
        align = torch.matmul(qrs, keys.t())#torch.exp(torch.matmul(qrs, keys.t())/0.07)
        top_k = torch.topk(align, 200, dim=1)
        scores = top_k.values
        indices = top_k.indices
        cls_k = self.oh[keys_cls[indices]]
        #cls_k = cls_k/cls_k.sum(dim=1, keepdim=True)
        pred_sc = (cls_k.to(self.device) * scores.unsqueeze(-1)).max(1).values
        return 1./(1e-6+torch.sqrt(2. - 2*pred_sc))
        pred = torch.max(pred_sc, dim=1)[1].cpu().tolist()
        score = f1_score(qrs_cls, pred, average='macro')
        return score
