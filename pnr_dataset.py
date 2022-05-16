from __future__ import absolute_import, division, print_function

import os
import sys
import h5py
import pickle
import numpy as np
import torch
import json
import copy
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

class PNRDataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.inference = mode == "test" 
        self.info = []
        self.init_info()
        
    def init_info(self):
        with open(self.opt.pnr_info,'r')as f:
            pnr_info = json.load(f)