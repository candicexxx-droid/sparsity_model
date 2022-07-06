import torch
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import models
import json
from utils import *
from tqdm import tqdm

# x = torch.tensor(1., requires_grad=True)
# z = torch.tensor(0., requires_grad=True)
# print('x:', x)
# y = x**2 + torch.log(z)
# print('y:', y)
# y.backward() # this is the same as y.backward(tensor(1.))
# print('x.grad:', x.grad)

def load_opt(path):
    
    # path='/Users/candicecai/Library/Mobile Documents/com~apple~CloudDocs/Desktop2/StarAI/sparsity_model/log/Jul04_13-22-42_id4'
    with open(path+"/hyperparam.json") as json_file:
        opt = json.load(json_file)
    
    return opt

def test(opt, path, test_likelihood=False):
    train_data, valid_data, test_data= DatasetFromFile(opt['data']),DatasetFromFile(opt['data'], 'valid'),DatasetFromFile(opt['data'],'test')
    
    if test_likelihood:
        train_data.unique(test_likelihood)
    
    train_dl, valid_dl, test_dl = DataLoader(train_data, batch_size=opt['batch_size']),DataLoader(valid_data, batch_size=opt['batch_size']),DataLoader(test_data, batch_size=opt['batch_size'])
    
    model = getattr(models, opt['model'])(train_data.info)
    chpt = torch.load(path+'/end_chpt.pt')
    model.load_state_dict(chpt['model'])

    outputs = []
    for i in tqdm(train_dl,leave=True):
        out = model(i)
        outputs.append(out)
    outputs = torch.cat(outputs)

    l = (torch.exp(outputs)).sum()
    print(opt['data'])
    print(l)
    return l


if __name__ == "__main__":
    path='/home/josh/testest/sparsity_model/log/Jul05_11-16-31_id0_arrayPC_ad'
    opt = load_opt(path)
    test(opt, path, True)
