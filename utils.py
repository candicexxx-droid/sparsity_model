import argparse
import sys
import os
from random import randint
import datetime
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from math import comb

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('-data', type=str, default='nips', help='specify dataset')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-epoch', default=50, type=int, help='epoch#')
    parser.add_argument('-cuda', type=int, default=0, help="specify cuda index")
    parser.add_argument('-group_num', type=int, default=1, help="group num for sum_arrayPCs")
    #optimizer
    parser.add_argument('-optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-weight_d', default=0, type=float)
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-model', type=str, default='LEnsemble', help='specify model name (same as class name in model.py')
    parser.add_argument('-output_folder', type=str, default="", help='log path')
    parser.add_argument('-output_dir', type=str, default="", help='log path') 
    parser.add_argument('-sanity_check', type=str, default='', help='sanity check, e.g 10,3; when this is non empty, data should be sanity_check')
    
    parser.add_argument('-EM_cluster_num', type=int, default=0, help='number of clusters; if 0 then no EM cluster')
    parser.add_argument('-EM_steps', type=int, default=0, help='number of EM steps')
    return parser.parse_args()

def process_opt(opt):
    rand_id='id'+str(randint(0,20))
    time_stamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

    if opt.output_folder:
        check_folder=os.path.join("log",opt.output_folder)
        if not os.path.isdir(check_folder):
            os.mkdir(check_folder)
            print('new folder %s created!'%check_folder)
           
    opt.output_dir= '_' + opt.output_dir if opt.output_dir else opt.output_dir
    opt.output_dir = time_stamp +'_' + rand_id + opt.output_dir

    opt.output_dir = os.path.join("log", opt.output_folder, opt.output_dir)
    
    
    print ("log will be saved at %s" % opt.output_dir)


def sum_n_k(n,k):
    """
    return sum_0<=i<=k(n,i)
    """
    total = 0
    for i in range(0,k+1):
        total+= comb(n,i)
    return total

def compute_usage(temp):
    d = np.unique(temp,axis=0,return_counts=True)[0]
    k = d.sum(axis=1).max()
    occur = d.shape[0]
    total_enum = sum_n_k(d.shape[1],k)
    usage = occur / total_enum

    return usage

def nll(y):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    ll = -torch.sum(y)
    return ll


class DatasetFromFile(Dataset):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/main/pgc/train.py
    """
    def __init__(self, dataset_name,  mode='train',x=None,group_num=1):
        """
        group_num deprecated, do not specify
        """
        if x is None:
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
            
            self.info={}
            self.info['k'] = int(x.sum(dim=1).max())
            self.info['n'] = x[0].shape[0]
        self.x = x
        
        self.splited = False
        if group_num>1:
            self.split_data_set(group_num)
            self.splited = True
            

    def __getitem__(self, index):
        if self.splited:
            ret = []
            for i in self.x:
                ret.append(i[index])

            return ret

        return self.x[index]

    def __len__(self):
        if self.splited:
            return self.x[0].shape[0]
        return len(self.x)
    
    # def print_info(self):
    #     print('============================================================================')
    #     print('%s size is %d'%(self.mode,len(self.x)) )
    #     print('shape of one obersvation:', self.x[0].shape[0])
    #     print('largest number of non zero entries in a single observation:', self.info['k'].item())
    #     print('============================================================================')
    
    def unique(self,test_likelihood=False):
        """
        return unique occurances of binary vector in the dataset
        """
        data = self.x.cpu().detach().numpy()
        if test_likelihood:
            self.x = torch.tensor(np.unique(data, axis=0))
        else:
            return np.unique(data, axis=0,return_counts=True)
    
    def split_data_set(self, group_num=1):
        """
        split and arrange data based on frequency of each unique appearance or duplicate
        input: DataFromFile object, group num
        output: return group_num of split data_set?
        """
        
        d = self.unique()[0].sum(axis=0)
        d = np.argsort(d)[::-1].copy() #with sorting
        # d = np.arange(self.info['n']) #no sorting
        idx_freq_by_var_sorted = torch.tensor(d) #in descending order
        self.x = self.x[:,idx_freq_by_var_sorted]
        splited = []
        step_size = math.ceil(self.info['n']/group_num)
        steps = list(range(0, self.info['n'],step_size))[1:]
        steps.append(self.info['n'])
        prev = 0
        self.info = []
        usages = []
        for i in steps:
            temp = self.x[:,prev:i]

            temp_info = {}
            splited.append(temp)
            temp_info['k'] = int(temp.sum(dim=1).max())
            temp_info['n'] = temp[0].shape[0]
            self.info.append(temp_info)
            # usage = compute_usage(temp) #check usage
            # usages.append(usage)
            prev = i
            # splited.append(temp)
            pass
        self.x = splited

        # print(sum(usages)/len(usages))









    


    


if __name__ == "__main__":
    data=DatasetFromFile('sanity_check')
    
    # data.split_data_set(group_num=5)
    # data[1]
    # test_l = DataLoader(data, batch_size=10)
    # for i in test_l:
    #     print(i)
    d = data.unique()
    gt_proportion = d[1]/data.x.shape[0]
    idea_ll = (np.log(gt_proportion)*d[1]).sum()/data.x.shape[0]
    # (np.log(gt_proportion) / (d[1]/data.x.shape[0])).sum() #perfect overfitting
    print('done')