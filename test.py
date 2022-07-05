import torch
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import models
import json
from utils import *

# x = torch.tensor(1., requires_grad=True)
# z = torch.tensor(0., requires_grad=True)
# print('x:', x)
# y = x**2 + torch.log(z)
# print('y:', y)
# y.backward() # this is the same as y.backward(tensor(1.))
# print('x.grad:', x.grad)


path='/Users/candicecai/Library/Mobile Documents/com~apple~CloudDocs/Desktop2/StarAI/sparsity_model/log/Jul04_20-26-57_id6'
# path='/Users/candicecai/Library/Mobile Documents/com~apple~CloudDocs/Desktop2/StarAI/sparsity_model/log/Jul04_13-22-42_id4'
with open(path+"/hyperparam.json") as json_file:
    opt = json.load(json_file)

train_data, valid_data, test_data= DatasetFromFile(opt['data']),DatasetFromFile(opt['data'], 'valid'),DatasetFromFile(opt['data'],'test')
train_dl, valid_dl, test_dl = DataLoader(train_data, batch_size=opt['batch_size']),DataLoader(valid_data, batch_size=opt['batch_size']),DataLoader(test_data, batch_size=opt['batch_size'])
model = getattr(models, opt['model'])(train_data.info)
chpt = torch.load(path+'/end_chpt.pt')

outputs = []
for i in train_dl:
    out = model(i)
    outputs.append(out)
outputs = torch.cat(outputs)

(torch.exp(outputs)).sum()