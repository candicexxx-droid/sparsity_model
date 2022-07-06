import argparse
from fileinput import filename
import sys
import os
from random import randint
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('-data', type=str, default='nips', help='specify dataset')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-epoch', default=50, type=int, help='epoch#')
    parser.add_argument('-cuda', type=int, default=0, help="specify cuda index")
    #optimizer
    parser.add_argument('-optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-weight_d', default=0, type=float)
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-model', type=str, default='LEnsemble', help='specify model name (same as class name in model.py')
    parser.add_argument('-output_folder', type=str, default="", help='log path')
    parser.add_argument('-output_dir', type=str, default="", help='log path')
    parser.add_argument('-sanity_check', type=str, default='', help='sanity check, e.g 10,3; when this is non empty, data should be sanity_check')
    return parser.parse_args()

def process_opt(opt):
    rand_id='id'+str(randint(0,9))
    if opt.output_folder:
        check_folder=os.path.join("log",opt.output_folder)
        if not os.path.isdir(check_folder):
            os.mkdir(check_folder)
            print('new folder %s created!'%check_folder)
            
    opt.output_dir= '_' + opt.output_dir if opt.output_dir else opt.output_dir
    opt.output_dir = datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_' +rand_id+opt.output_dir

    opt.output_dir = os.path.join("log", opt.output_folder, opt.output_dir)
    
    
    print ("log will be saved at %s" % opt.output_dir)



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
                if 'mnist' in dataset_name:
                    line = [int(x) for x in line.split(' ')]
                else:
                    line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.tensor(examples,dtype=torch.long)
        self.x = x
        self.info={}
        self.info['k'] = int(x.sum(dim=1).max())
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
    
    def unique(self,test_likelihood=False):
        """
        return unique occurances of binary vector in the dataset
        """
        data = self.x.cpu().detach().numpy()
        if test_likelihood:
            self.x = torch.tensor(np.unique(data, axis=0))
        else:
            return np.unique(data, axis=0)

if __name__ == "__main__":
    data=DatasetFromFile('ad')
    d = data.unique()
    print('done')