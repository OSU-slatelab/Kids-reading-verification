from model2 import *
from util import *
from data import *
from knn import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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

def mhot_tgt(tlist, ncls):
    tgt = torch.zeros(len(tlist), ncls)
    for i,t in enumerate(tlist):
        tgt[i][t] = 1.
    return tgt

class SupContrastiveLoss(nn.Module):
    def __init__(self, device, temp=0.07):
        super(SupContrastiveLoss, self).__init__()
        self.device = device
        self.temp = temp 

    def forward(self, r, label):
        MASK = torch.eye(label.size(0)).to(self.device)
        PEN = -10000000.

        r = F.normalize(r, dim=1)
        align_r = torch.log_softmax((torch.matmul(r, r.t()) / self.temp) + MASK*PEN, dim=1)

        label = F.normalize(label, dim=1)
        align_label = torch.matmul(label, label.t()) * (1. - MASK)
        align_label = align_label / align_label.sum(dim=1, keepdim=True)
        align_label[align_label != align_label] = 0.
        
        loss = -1. * self.temp * (align_r * align_label).sum(dim=1).mean()
        return loss

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer=None, data_valid=None, data_test=None):
        self.args = args
        self.data = data
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 27), time_mask_width=(0, 100))
        if self.args.decouple:
            collator = CollatorDec(args)
        elif self.args.word_by_word:
            collator = WordCollator(args)
        elif self.args.phonet:
            collator = PhonetCollator(args)
        else:
            collator = Collator(args)
        
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collator)
        self.optimizer = optimizer
        if self.args.pretrain:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, total_steps=args.nsteps, anneal_strategy='linear', pct_start=0.1)
        self.loader_va = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collator)
        self.loader_te = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collator)
        self.loader_map = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=8, collate_fn=collator)
        #self.cls_loss_cb = CBCrossEntropy([3500, 30, 110, 50, 27, 15, 1], device, beta=args.beta, gamma=args.gamma) #nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.cls_loss = nn.CrossEntropyLoss()
        self.f_loss = FScore()
        self.con_loss = SupContrastiveLoss(device)
        self.device = device
        self.norm = InputNormalization(update_until_epoch=args.norm_epoch)
        if self.args.normalizer != '':
            self.norm = load_pick(self.args.normalizer)
        self.knn_moniter = kNN(collator, self.loader, device, self.norm, bsz=4, ncls=7)

    def opt_step(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer.step()

    def opt_step2(self, model):
        nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

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

    def evaluate(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(loader):
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
            if self.args.pretrain:
                label = text_tgt
            else:
                label = cls_tgt
            with torch.no_grad():
                pred, _, _, _, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.tolist())
        if self.args.pretrain:
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, average='macro')#'binary')
            if test:
                return score, confusion_matrix(y_true, y_pred)
        return score

    def evaluate_word(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens_s, lmax, cls_tgt in tqdm(loader):
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            sbatch = load2gpu(sbatch, self.device)
            label = cls_tgt
            with torch.no_grad():
                pred = model(sbatch, lens_s)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.tolist())
        score = f1_score(y_true, y_pred, average='binary')#'binary')
        if test:
            return score, confusion_matrix(y_true, y_pred)
        return score

    def evaluate_phonet(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        for X, X_post, X_phon, lens_mfcc, lmax, lens_post, lens_ph, cls_tgt in tqdm(loader):
            lens_norm = lens_mfcc / lmax
            X_mfcc = self.norm(X, lens_norm.float(), epoch=1000)
            X_mfcc = load2gpu(X_mfcc, self.device)
            X_post = load2gpu(X_post, self.device)
            X_phon = load2gpu(X_phon, self.device)
            label = cls_tgt
            with torch.no_grad():
                pred = model(X_mfcc, X_post, X_phon, lens_mfcc, lens_post, lens_ph)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.tolist())
        score = f1_score(y_true, y_pred, average='binary')#'macro')
        if test:
            return score, confusion_matrix(y_true, y_pred)
        return score

    def evaluate_knn(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        pred_bat = []
        label_bat = []
        for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(loader):
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
            if self.args.pretrain:
                label = text_tgt
            else:
                label = cls_tgt
            with torch.no_grad():
                pred, _, _, _, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)
            pred = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True)+1e-6)
            pred_bat.append(pred)
            label_bat.append(label)
        labels = torch.cat(label_bat, dim=0)
        pred_cls = torch.cat(pred_bat, dim=0)
        pred_knn = self.knn_moniter.classify(model, loader)
        pred_knn = (pred_knn - pred_knn.mean(dim=1, keepdim=True)) / (pred_knn.std(dim=1, keepdim=True) + 1e-6)
        pred = 0.25*pred_knn + 0.75*pred_cls
        y_pred = torch.max(pred, dim=1)[1].cpu().tolist()
        y_true = labels.tolist()
        if self.args.pretrain:
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, average='macro')#'binary')
            if test:
                return score, confusion_matrix(y_true, y_pred)
        return score

    def evaluate_decouple(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cat_tgt, det_tgt in tqdm(loader):
            lens_norm = lens_s / lmax
            sbatch = self.norm(X, lens_norm.float(), epoch=1000)
            sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
            with torch.no_grad():
                pred_cat, pred_det, _ = model.decouple_hier(sbatch, tbatch, lens_s, lens_t, merge_idx, label_cat=cat_tgt, label_det=det_tgt)
            y_true.extend(cat_tgt.tolist())
            #y_true.extend(det_tgt.tolist())
            y_det = torch.max(pred_det, dim=1)[1].cpu().tolist()
            #y_pred.extend(y_det)
            for i, detection in enumerate(y_det):
                if detection == 1:
                    y_pred.append(torch.argmax(pred_cat[i][1:]).item() + 1)
                else:
                    y_pred.append(0)
        score = f1_score(y_true, y_pred, average='macro')#'binary')
        #score = f1_score(y_true, y_pred, average='binary')
        if test:
            return score, confusion_matrix(y_true, y_pred)
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
                loss.backward()
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
            #save(model, self.args.save_path+'_last.pt')
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

    def detect_word(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        rpat = self.args.patience
        steps = 0
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens_s, lmax, cls_tgt in tqdm(self.loader):
                steps += 1
                model.train()
                lens_norm = lens_s / lmax
                sbatch = self.norm(X, lens_norm.float(), epoch=epoch-1)
                sbatch = load2gpu(sbatch, self.device)
                label_cls = load2gpu(cls_tgt, self.device)
                pred_cls = model(sbatch, lens_s)#, label=label_cls)
                loss = self.cls_loss(pred_cls, label_cls)
                self.opt_step(model, loss)
                loss_list.append(loss.item())
            print(f'Running validation.') 
            score_val = self.evaluate_word(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
        print(f'Running test.') 
        score_test, conf = self.evaluate_word(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        print(conf)
        logger.info(f'| test score = {score_test} |')
        logger.info(conf)
        
    def detect_phonet(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        rpat = self.args.patience
        steps = 0
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, X_post, X_phon, lens_mfcc, lmax, lens_post, lens_ph, cls_tgt in tqdm(self.loader):
                steps += 1
                model.train()
                lens_norm = lens_mfcc / lmax
                X_mfcc = self.norm(X, lens_norm.float(), epoch=epoch-1)
                X_mfcc = load2gpu(X_mfcc, self.device)
                label_cls = load2gpu(cls_tgt, self.device)
                X_post = load2gpu(X_post, self.device)
                X_phon = load2gpu(X_phon, self.device)
                pred_cls = model(X_mfcc, X_post, X_phon, lens_mfcc, lens_post, lens_ph)
                loss = self.cls_loss(pred_cls, label_cls)
                self.opt_step(model, loss)
                loss_list.append(loss.item())
            print(f'Running validation.') 
            score_val = self.evaluate_phonet(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
        print(f'Running test.') 
        score_test, conf = self.evaluate_phonet(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        print(conf)
        logger.info(f'| test score = {score_test} |')
        logger.info(conf)

    def detect(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        loss_cls_list = []
        loss_tok_list = []
        rpat = self.args.patience
        steps = 0
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cls_tgt in tqdm(self.loader):
                steps += 1
                model.train()
                lens_norm = lens_s / lmax
                sbatch = self.norm(X, lens_norm.float(), epoch=epoch-1)
                sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
                label_cls = load2gpu(cls_tgt, self.device)
                label_tok = load2gpu(text_tgt, self.device)
                pred_cls, pred_tok, _, _, _ = model(sbatch, tbatch, lens_s, lens_t, merge_idx)#, label=label_cls)
                loss_cls = self.cls_loss(pred_cls, label_cls)
                loss_tok = self.cls_loss(pred_tok, label_tok)
                loss = 0.5*loss_cls + 0.5*loss_tok
                self.opt_step(model, loss)
                loss_list.append(loss.item())
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
            log = f'| epoch = {epoch} | loss_cls = {np.mean(loss_cls_list)} | loss_tok = {np.mean(loss_tok_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
            loss_cls_list = []
            loss_tok_list = []
        print(f'Running test.') 
        score_test, conf = self.evaluate(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        print(conf)
        logger.info(f'| test score = {score_test} |')
        logger.info(conf)

    def detect_pairwise(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
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
                pred_cls, pred_tok, rep, _, label_mix = model(sbatch, tbatch, lens_s, lens_t, merge_idx, label=label_cls)
                label_oh = mhot_tgt(label_cls.tolist(), 7).to(self.device)
                loss_con = self.con_loss(rep, label_oh)
                loss_cls = self.cls_loss(pred_cls, label_cls)#self.kld_loss(torch.log_softmax(pred_cls,dim=1), label_mix)#self.cls_loss(pred_cls, label_cls)
                loss_tok = self.cls_loss(pred_tok, label_tok)
                loss = 0.15*loss_con + 0.85*loss_cls + loss_tok #0.5*loss_cls + 0.5*loss_tok
                self.opt_step(model, loss)
                loss_list.append(loss.item())
                loss_cls_list.append(loss_cls.item())
                loss_tok_list.append(loss_tok.item())
            print(f'Running validation.') 
            score_val = self.evaluate_knn(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss_cls = {np.mean(loss_cls_list)} | loss_tok = {np.mean(loss_tok_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
            loss_cls_list = []
            loss_tok_list = []
        print(f'Running test.') 
        score_test, conf = self.evaluate_knn(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        save(best_model, f'{self.args.save_path}.pt')
        print(conf)
        logger.info(f'| test score = {score_test} |')
        logger.info(conf)

    def detect_decouple(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        loss_cls_list = []
        loss_tok_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens_s, lmax, tbatch, lens_t, text_tgt, merge_idx, cat_tgt, det_tgt in tqdm(self.loader):
                model.train()
                lens_norm = lens_s / lmax
                sbatch = self.norm(X, lens_norm.float(), epoch=epoch-1)
                sbatch, tbatch = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device)
                label_cat = load2gpu(cat_tgt, self.device)
                label_det = load2gpu(det_tgt, self.device)
                label_tok = load2gpu(text_tgt, self.device)
                pred_cat, pred_det, pred_tok = model.decouple_hier(sbatch, tbatch, lens_s, lens_t, merge_idx, label_cat=label_cat, label_det=label_det)
                ###
                where_0 = (label_cat == 0).nonzero(as_tuple=True)[0].tolist()
                idx = []
                for i in range(label_cat.size(0)):
                    if i not in where_0:
                        idx.append(i)
                pred_cat = pred_cat[idx]
                label_cat = label_cat[idx]
                ###
                if label_cat.size(0) > 0:
                    loss_cat = self.cls_loss(pred_cat, label_cat)
                else:
                    loss_cat = torch.tensor(0.)
                loss_det = self.cls_loss(pred_det, label_det)
                loss_tok = self.cls_loss(pred_tok, label_tok)
                loss_cls = loss_cat + loss_det
                loss = 0.5*loss_cls + 0.5*loss_tok
                self.opt_step(model, loss)
                loss_list.append(loss.item())
                loss_cls_list.append(loss_cls.item())
                loss_tok_list.append(loss_tok.item())
            save_pick(self.norm, '/data/data24/scratch/sunderv/saved_models/readr.pickle')
            print(f'Running validation.') 
            score_val = self.evaluate_decouple(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss_cls = {np.mean(loss_cls_list)} | loss_tok = {np.mean(loss_tok_list)} | dev_f1 = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
            loss_cls_list = []
            loss_tok_list = []
        print(f'Running test.') 
        score_test, conf = self.evaluate_decouple(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        print(conf)
        logger.info(f'| test score = {score_test} |')
        logger.info(conf)
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
