from model2 import *
from util import *
from data import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.augment import SpecAugment
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def euc_loss(r1, r2):
    r1 = F.normalize(r1, dim=1)
    r2 = F.normalize(r2, dim=1)
    return (2. - 2. * ((r1*r2).sum(dim=1))).mean()

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

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer=None, data_valid=None, data_test=None):
        self.args = args
        self.data = data
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 27), time_mask_width=(0, 100))
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collator)
        self.optimizer = optimizer
        if self.args.pretrain:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, total_steps=args.nsteps, anneal_strategy='linear', pct_start=0.1)
        self.loader_va = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collator)
        self.loader_te = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collator)
        self.loader_map = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=8, collate_fn=collator)
        self.cls_loss = nn.CrossEntropyLoss()
        self.device = device
        self.norm = InputNormalization(update_until_epoch=args.norm_epoch)
        if self.args.normalizer != '':
            self.norm = load_pick(self.args.normalizer)

    def opt_step(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer.step()

    def get_attn_map(self, steps, model, loader, path='attn_maps/lib100_las_'):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens, lmax, tbatch, text_tgt, merge_idx, cls_tgt in loader:
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=1000)
            sbatch = list_batch(sbatch, lens, lmax)
            sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
            if self.args.pretrain:
                label = load2gpu(text_tgt, self.device)
            else:
                label = load2gpu(cls_tgt, self.device)
            with torch.no_grad():
                pred, attn_tq = model(sbatch, tbatch, merge_idx)
            attn_tq = attn_tq[0].detach().cpu().numpy()
            np.save(f'{path}_{steps}.npy', attn_tq)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.cpu().tolist())
            break

    def evaluate(self, model, loader):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(loader):
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            if self.args.pretrain:
                label = load2gpu(text_tgt, self.device)
            else:
                label = load2gpu(cls_tgt, self.device)
            with torch.no_grad():
                pred, _, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.cpu().tolist())
        if self.args.pretrain:
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, average='binary')
        return score

    def pretrain(self, model, logger, nsteps, log_after, val_after):
        steps = self.args.steps_done
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        scores_val = []
        flag = False
        while True:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(self.loader):
                model.train()
                steps += 1
                lens_norm = lens_s / lmax
                sbatch = self.norm(X, lens_norm.float(), epoch=epoch-1)
                sbatch = self.aug(sbatch)  #TODO adding specaugment
                sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
                label = load2gpu(text_tgt, self.device)
                pred, _, _ = model(sbatch, tbatch, lens_s, lens_t)
                loss = self.cls_loss(pred, label)
                self.opt_step(model, loss)
                self.scheduler.step()
                loss_list.append(loss.item())
                if steps % log_after == 0:
                    log = f'| steps = {steps} | loss = {np.mean(loss_list)} |'
                    logger.info(log)
                    loss_list = []
                if steps % val_after == 0:
                    print(f'Running validation.') 
                    score_val = self.evaluate(model, self.loader_va)
                    scores_val.append(score_val)
                    np.save(f'dev_scores/{self.args.logging_file[5:-4]}.npy', np.array(scores_val))
                    if best_score is None or best_score < score_val:
                        best_model = copy.deepcopy(model)
                        best_score = score_val
                    log = f'| steps = {steps} | dev_acc = {score_val} |'
                    logger.info(log)
                    #self.get_attn_map(steps, model, self.loader_map, path='attn_maps/lib100pt_racept')
                if steps % self.args.save_after == 0:
                    save(model, f'{self.args.save_path}_steps_{steps}.pt')
                if steps % nsteps == 0:
                    flag = True
                    break
            if flag:
                break
        print(f'Running test.') 
        score_test = self.evaluate(best_model, self.loader_te)
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
            save(model, self.args.save_path+'_last.pt')
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

    def detect(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_cls_list = []
        loss_tok_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(self.loader):
                model.train()
                lens_norm = lens_s / lmax
                sbatch = self.norm(X, lens_norm.float(), epoch=epoch-1)
                sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
                label_cls = load2gpu(cls_tgt, self.device)
                label_tok = load2gpu(text_tgt, self.device)
                pred_cls, pred_tok, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)
                loss_cls = self.cls_loss(pred_cls, label_cls)
                loss_tok = self.cls_loss(pred_tok, label_tok)
                loss = (1. - self.args.asr_wt)*loss_cls + self.args.asr_wt*loss_tok
                self.opt_step(model, loss)
                loss_cls_list.append(loss_cls.item())
                loss_tok_list.append(loss_tok.item())
            print(f'Running validation.') 
            score_val = self.evaluate(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss_cls = {np.mean(loss_cls_list)} | loss_asr = {np.mean(loss_tok_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_cls_list = []
            loss_tok_list = []
        print(f'Running test.') 
        score_test = self.evaluate(best_model, self.loader_te)
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

