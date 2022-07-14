import torch
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import models
import json
from utils import *
from tqdm import tqdm
import argparse
from train import avg_ll

# x = torch.tensor(1., requires_grad=True)
# z = torch.tensor(0., requires_grad=True)
# print('x:', x)
# y = x**2 + torch.log(z)
# print('y:', y)
# y.backward() # this is the same as y.backward(tensor(1.))
# print('x.grad:', x.grad)

def parse_args():
    parser = argparse.ArgumentParser("test")
    parser.add_argument('-path', type=str, default='/space/candicecai/sparsity_model/log/Jul11_22-47-50_id10_arrayPC-bbc', help='specify dataset')
    parser.add_argument('-test_likelihood',action='store_true', default=False,help='test the total likelihood of unique occurence in the train dataset (see the extent of likelihood maximization')
    parser.add_argument('-cuda', type=int, default=0, help="specify cuda index")
    return parser.parse_args()

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
    device_name="cuda:%d"%opt['cuda'] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = model.to(device)
    test_ll = avg_ll(model, test_dl,device)
    print(test_ll)
    # outputs = []
    # for i in tqdm(test_dl,leave=True):
    #     out = model(i)
    #     outputs.append(out)
    # outputs = torch.cat(outputs)

    # l = (torch.exp(outputs)).sum()
    # print(opt['data'])
    # print(l)
    # return l



if __name__ == "__main__":
    args = parse_args()
    opt = load_opt(args.path)
    test(opt, args.path, args.test_likelihood)
