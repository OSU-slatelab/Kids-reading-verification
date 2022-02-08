from util import *
from model2 import *
from train import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
import torch.optim as optim
import torch
import pdb
import logging
import copy
import argparse
import time
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='')
    parser.add_argument('--pretrain', action='store_true',
                        help='')
    parser.add_argument('--best-model', action='store_true',
                        help='')
    parser.add_argument('--save-model', action='store_true',
                        help='')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--slu-data', type=str, default='hvb',
                        help='')
    parser.add_argument('--normalizer', type=str, default='',
                        help='')
    parser.add_argument('--logging-file', type=str, default='',
                        help='log file')
    parser.add_argument('--tokenizer-path', type=str, default='',
                        help='tokenizer file json')
    parser.add_argument('--train-path', type=str, default='',
                        help='training file')
    parser.add_argument('--valid-path', type=str, default='',
                        help='validation file')
    parser.add_argument('--test-path', type=str, default='',
                        help='testing file')
    parser.add_argument('--audio-path', type=str, default='',
                        help='where speech files are saved')
    parser.add_argument('--save-path', type=str, default='',
                        help='')
    parser.add_argument('--dict-path', type=str, default='',
                        help='')
    parser.add_argument('--embed-dim', type=int, default=320,
                        help='')
    parser.add_argument('--nspeech-feat', type=int, default=80,
                        help='')
    parser.add_argument('--pyr-layer', type=int, default=3,
                        help='')
    parser.add_argument('--nlayer', type=int, default=6,
                        help='')
    parser.add_argument('--nhead', type=int, default=1,
                        help='')
    parser.add_argument('--nsteps', type=int, default=250000,
                        help='')
    parser.add_argument('--steps-done', type=int, default=0,
                        help='')
    parser.add_argument('--log-after', type=int, default=100,
                        help='')
    parser.add_argument('--val-after', type=int, default=500,
                        help='')
    parser.add_argument('--save-after', type=int, default=500,
                        help='')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='')
    parser.add_argument('--nclasses', type=int, default=16,
                        help='')
    parser.add_argument('--patience', type=int, default=16,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='')
    parser.add_argument('--asr-wt', type=float, default=0.1,
                        help='')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='')
    parser.add_argument('--norm-epoch', type=int, default=3,
                        help='')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='')

    args = parser.parse_args()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda:0")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) # ignored if not --cuda
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rfh = RotatingFileHandler(args.logging_file, maxBytes=100000, backupCount=10, encoding="UTF-8")
    logger.addHandler(rfh)

    vocab_size = 2+Tokenizer.from_file(args.tokenizer_path).get_vocab_size()
    config = {'embed_dim':args.embed_dim, 'vocab_size':vocab_size, 'dropout':args.dropout, 'input_dim':args.nspeech_feat, 'pyr_layer':args.pyr_layer, 'nlayer':args.nlayer, 'multi-gpu':args.multi_gpu, 'nhead':args.nhead, 'nclasses':args.nclasses, 'pretrain':args.pretrain}

    print(f'Loading model.')
    if args.pretrain:
        model = ASR(config)
    else:
        model = Detector(config)
    if args.dict_path != '':
        load_dict(model, args.dict_path) 
    model = model.to(device)
    print(f'Done.')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    data_train = SpeechDataset(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    data_valid = SpeechDataset(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    data_test = SpeechDataset(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    trainer = Trainer(args, data_train, device, optimizer, data_valid=data_valid, data_test=data_test)
    if args.pretrain:
        trainer.pretrain(model, logger, args.nsteps, args.log_after, args.val_after)
    else:
        trainer.detect(model, logger)
