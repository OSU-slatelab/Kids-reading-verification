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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='',
                        help='dictionary in str form (from json.dumps)')
    tokenizer_path = "tokenizers/librispeech.json"
    device = torch.device("cpu")
