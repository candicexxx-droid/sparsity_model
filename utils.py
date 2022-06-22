import argparse
from fileinput import filename
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('-data', type=str, default='nips', help='specify dataset')
    return parser.parse_args()




class DatasetFromFile(Dataset):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/main/pgc/train.py
    """
    def __init__(self, dataset_name, mode='train'):
        examples = []
        self.mode = mode
        #run init under project root
        filename = 'data/'+dataset_name+'/'+dataset_name+'.'+mode+'.data'
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.LongTensor(examples)
        self.x = x
        self.info={}
        self.info['k'] = x.sum(dim=1).max()
        self.info['n'] = self.x[0].shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)
    
    def print_info(self):
        print('============================================================================')
        print('%s size is %d'%(self.mode,len(self.x)) )
        print('shape of one obersvation:', self.x[0].shape[0])
        print('largest number of non zero entries in a single observation:', self.info['k'].item())
        print('============================================================================')

if __name__ == "__main__":
    data=DatasetFromFile('nips')
    data.print_info()
    print('done')