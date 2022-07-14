from tokenize import group
from turtle import forward
from numpy import array
import torch
import torch.nn as nn
import itertools


#see how to pass children's result to parent
class sumNode(nn.Module):
    def __init__(self, parent, children, leaf=False) -> None:
        super().__init__()
        self._children = children #list
        num_child = len(children) if type(self._children) is not int else children
        self.W = nn.Parameter(torch.rand(num_child,1), requires_grad=True)
        self.parent = parent #single node 
        
        self.val = None
        self.leaf=leaf
    def forward(self,x=None):
        """
        x should be in size num_child x 1
        """
        if not self.leaf: #if sum node is not leaf
            out = []
            for i in self._children:
                out.append(i(x)) #summ all outputs of children  together 
            out = torch.cat(out,dim=1)
            
        else: #'leaf sum node'
            #+
            #x_1,...,x_n (in PGC form) -> computed in PC version            
            out=[]
            l = len(self._children)
            
            for i in range(l):
                temp = (x^True)[:,:l]
                temp[:,i]=1

                out.append(x[:,i]*torch.prod(temp,dim=1))

            out = torch.stack(out,dim=1)
        W = nn.functional.softmax(self.W.transpose(1,0)).transpose(1,0)
        out = torch.matmul(out.to(torch.float),W)
        
        return out 

class prodNode(nn.Module):
    def __init__(self, parent, children) -> None:
        super().__init__()
        self.parent = parent #single node 
        self._children = children #list or index to input 
        self.val = None
    def forward(self,x=None):
        
        out = []
        for i in self._children:
            out.append(i(x))
        # out = torch.as_tensor(out)
        out = torch.cat(out,dim=1)
       
        
        out = torch.prod(out, dim=1, keepdim=True)
        return out

class inputNode(nn.Module):
    def __init__(self,data_idx) -> None:
        super().__init__()
        self.idx=data_idx
    
    def forward(self,x ):
        return torch.unsqueeze(x[:,self.idx],dim=1).to(torch.float)

class naivePC(nn.Module):
    def __init__(self, info) -> None:
        '''
        info: k and ob_l from data
        model probability distribution of exact k nonzero terms (time complexity is too large)
        '''
        super().__init__()
        self.info=info #get n and k 
        self.root= sumNode(None, self._growPC(info))
    

    def _growPC(self,info):
        if info['k']==info['n']:

            children=[]
            for i in range(info['k']):
                children.append(inputNode(i))

            prod_n=prodNode(None,children)
            return [prod_n]
        if info['k']==1:
            children=[]
            for i in range(info['n']):
                children.append(inputNode(i))
            sum_n=sumNode(None,children,True)
            return [sum_n]
        info1={}
        info2={}


        info1['k']=info['k']
        info1['n']=info['n']-1 

        info2['k']=info['k']-1
        info2['n']=info['n']-1 

        left_tree=self._growPC(info1)[0] if info1['k']==info1['n'] else sumNode(None, self._growPC(info1))
        # right_tree=prodNode(None, self._growPC(info2)+ [inputNode(info['n']-1)]) #right tree must have 2 children 
        right_tree=prodNode(None,  [sumNode(None, self._growPC(info2)),inputNode(info['n']-1)]) 

        return [left_tree,right_tree]
        # return [self._growPC(info1), prodNode(None, self._growPC(info2)+ [inputNode(info['k']-1)])]

    def forward(self,x):

        return self.root(x)

class bruteForce(nn.Module):
    def __init__(self, data_info) -> None:
        super().__init__()
        
        self.n = data_info['n']
        seq = list(range(self.n))
        total_case = 0
        self.k = data_info['k']
        self.table = []

        for i in range(self.k+1):
            comb = list(itertools.combinations(seq,i))
            for j in comb:
                t = torch.zeros(self.n)
                idx = torch.tensor(j)
                if len(idx)>0:
                    t[idx] = 1 
                self.table.append(t)



            total_case += len(comb)
        self.table = torch.stack(self.table).to(torch.int)
        self.W = nn.Parameter(torch.randn(total_case),requires_grad=True)
        print(' ')



    def forward(self,x):
        out = []
        W = nn.functional.softmax(self.W)
        for i in x:
            check_eql = (torch.sum(self.table^i, axis=1)==0).nonzero()[0]
            out.append(W[check_eql])
        out = torch.stack(out)
        return torch.log(out)



        pass 


class LEnsemble(nn.Module):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc
    """

    def __init__(self, data_info) -> None:
        super().__init__()
        self.n = data_info['n']
        n = self.n
        # k = data_info['k']
        # # B = torch.randn(n, data_info['k'])
        # B = torch.randn(n, n)
        # B_norm = torch.norm(B, dim=0)
        # for i in range(0, k):
        #     B[:,i] /= B_norm[i]
        # self.B = nn.Parameter(B, requires_grad=True) #n *k 
        # self.I = torch.eye(n)
        
        # self.n = n

        k = data_info['k']
        B = torch.randn(n, data_info['k'])
        B_norm = torch.norm(B, dim=0)
        for i in range(0, k):
            B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

        

    def forward(self,x): #output log likelihood
        # n = self.n
        # batch_size = x.shape[0]

        # eps = 1e-8
        # I = self.I.to(x.device)
        # L = torch.matmul(self.B,self.B.transpose(1,0))  + eps * I
        # L0 = L.clone()
        # L = L.unsqueeze(0).repeat(batch_size, 1, 1)

        # L[x == 0] = 0.0
        # L[x.unsqueeze(1).repeat(1,n,1) == 0] = 0.0
        # L[torch.diag_embed(1-x) == 1] = 1.0

        # y = torch.logdet(L)
        # if torch.isnan(y).sum()>0:
        #     print('find nan!')
        # return y - torch.logdet(L0 + I)
        n = self.n
        batch_size = x.shape[0]

        eps = 1e-8
        I = torch.eye(n).to(x.device)
        L = torch.matmul(self.B,self.B.transpose(1,0))  
        # + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)

        L[x == 0] = 0.0
        L[x.unsqueeze(1).repeat(1,n,1) == 0] = 0.0
        L[torch.diag_embed(1-x) == 1] = 1.0

        y = torch.logdet(L)
        return y - torch.logdet(L0 + I)


class compElemDPP(nn.Module):
    def __init__(self, data_info) -> None:
        super().__init__()
        self.n = data_info['n']
        self.k = data_info['k']
        B = torch.rand(self.n, self.n)
        # B_norm = torch.norm(B, dim=0)
        # for i in range(0, self.n):
        #     B[:,i] /= B_norm[i]#make B orthonormal
        self.B = nn.Parameter(B, requires_grad=True)
        self.I = torch.eye(self.n)

    def forward(self,x):

        L = torch.matmul(self.B.transpose(0,1),self.B)
        _, V = torch.linalg.eigh(L)
        V_norm = torch.norm(V, dim=0)
        for i in range(0, self.n):
            V[:,i] /= V_norm[i]#make V orthonormal
        


        L_ens = []
        for i in range(1, self.k+1):
            #construct marginal kernels for elementary DPP 
            K_i = torch.matmul(V[:,:i], V[:,:i].transpose(1,0))
            L_i = torch.matmul(K_i,torch.linalg.inv(self.I-K_i)) #entries of L_i explodes here because entries of K_i is very small 
            L_ens.append(L_i)
        
        L_ens = torch.stack(L_ens)
        # k = x.sum(dim=1) #compute number of nonzero terms for each example in a batch
        # k = k-1 #make it zero-index
        # L_ens_x = L_ens[k] 
        outputs = []
        for i in x:
            idx = (i==1)
            k = i.sum().item()-1 ##compute number of nonzero terms for each example in a batch, make it zero-indexed
            L_k = L_ens[k]
            L_y = L_k[idx,:][:,idx]
            log_norm = torch.logdet(L_k+self.I)
            out = torch.logdet(L_y)-log_norm # log likelihood
            outputs.append(out)
        outputs = torch.stack(outputs)
        return outputs #output log likelihood for each example

class arrayPC(nn.Module):
    def __init__(self, data_info)-> None:
        super().__init__()
        self.n, self.k = data_info['n'],(data_info['k']+1)

        self.W = nn.Parameter(torch.randn(self.n-1,self.k-1,2), requires_grad=True)
        self.endW = nn.Parameter(torch.randn(1,self.k), requires_grad=True)
        self.W_adjust1 = torch.ones(self.W.shape)
        shape = self.W_adjust1[0].shape
        self.W_adjust2 = torch.zeros(self.W.shape)
        for i in range(0, self.k-2):
            temp = self.W_adjust1[i].view(-1)
            temp[2+2*i:] = 0
            self.W_adjust1[i] = temp.view(shape)
            temp = self.W_adjust2[i].view(-1)
            temp[2+2*i+1] = 1
            self.W_adjust2[i] = temp.view(shape)

        self.episl = 1e-15
    def forward(self, x):
        W = nn.functional.softmax(self.W,dim=2)
        W_append = torch.zeros(self.n-1,1,2)
        W_append[:,:,0] = 1
        W_full = W * self.W_adjust1 + self.W_adjust2
        W_full = torch.cat([W_append,W_full],dim = 1)
        group_num = torch.clone(x[:,0])
        out = torch.zeros(x.shape[0],1)
        for i in range(1,self.n):
            group_num+=x[:,i]
            idx = x[:,i]
            selected_group = torch.index_select(W_full[i-1],0,group_num)
            # print('hi')
            out +=torch.log(selected_group.gather(1,idx.unsqueeze(1)))
            pass
        endW = nn.functional.softmax(self.endW,dim=1)
        endW = torch.index_select(endW,1,group_num).transpose(1,0)
        out = out + torch.log(endW)
        return out

class multi_arrayPC(nn.Module): #a prod node
    def __init__(self, data_info) -> None:
        super().__init__()
        self.array_PCs = []
        for i in data_info:
            self.array_PCs.append(arrayPC(i))
        # print(' ')
    def to(self,device):
        self.array_PCs = [i.to(device) for i in self.array_PCs]
        return self
    def forward(self,x):
        outputs = []
        for i in range(len(x)):
            outputs.append(self.array_PCs[i](x[i]))
        # print(" ")
        out = torch.cat(outputs).sum()
        return out

class sum_arrayPCs(nn.Module): #a prod node
    def __init__(self, data_info, group_num=10) -> None:
        super().__init__()
        self.array_PCs = []
        self.group_num = group_num
        for i in range(group_num):
            self.array_PCs.append(arrayPC(data_info))
        # print(' ')
        self.W = nn.Parameter(torch.randn(1,group_num), requires_grad=True)
    def to(self,device):

        self.array_PCs = [i.to(device) for i in self.array_PCs]
        return self
    def forward(self,x):
        outputs = []
        for i in range(self.group_num):
            outputs.append(self.array_PCs[i](x))
        # print(" ")
        W = nn.functional.softmax(self.W,dim=1).to(x.device)
        out = torch.matmul(W,torch.stack(outputs))[0]
        return out




class arrayPC_naive(nn.Module):
    def __init__(self, data_info)-> None:
        super().__init__()
        self.n, self.k = data_info['n'],(data_info['k']+1)
        self.W = nn.Parameter(torch.randn(self.n-1,self.k-1,2), requires_grad=True)
        self.endW = nn.Parameter(torch.randn(1,self.k), requires_grad=True)
        self.W_adjust1 = torch.ones(self.W.shape)
        shape = self.W_adjust1[0].shape
        self.W_adjust2 = torch.zeros(self.W.shape)
        for i in range(0, self.k-2):
            temp = self.W_adjust1[i].view(-1)
            temp[2+2*i:] = 0
            self.W_adjust1[i] = temp.view(shape)
            temp = self.W_adjust2[i].view(-1)
            temp[2+2*i] = 1
            self.W_adjust2[i] = temp.view(shape)
        self.episl = 1e-15
        self.endW.data = self.endW.data.transpose(1,0)
    def forward(self, x):
         #x.shape = B, n
        F = torch.zeros(x.shape[0], self.n, self.k)
        # base case
        F[:,:,0] = 1 
        F[:,0,1] = x[:,0] #x_1
        
        W = nn.functional.softmax(self.W,dim=2)
        W_full = W * self.W_adjust1 + self.W_adjust2
        for i in range(0,self.n):
            for j in range (0, i+1):
                F[:,i,0] *= x[:,j]^True
        # print('base case done')
        for i in range(1, self.n):
            W = W_full[i-1]
            F[:,i,1:] = x[:,i].unsqueeze(1)*W[:,0]*F[:,i-1,:self.k-1].clone() + (x[:,i]^True).unsqueeze(1)*W[:,1]*F[:,i-1,1:].clone() #
            pass 
        endW = nn.functional.softmax(self.endW,dim=0)
        out = torch.matmul(F[:,-1,:],endW)
        out = torch.log(out)
        return out

            





if __name__=="__main__":
    from utils import *
    from sanity_check_gen import *
    from torch.utils.data import DataLoader
    import models
    from time import time
    from tqdm import tqdm
    torch.manual_seed(0)

    sanity_check_gen(5,3)
    opt=parse_args()
    train_data=DatasetFromFile(opt.data)
    train_dl = DataLoader(train_data, batch_size=7)
    # train_data.info['k']=2
    # start_time = time()
    # model=naivePC(train_data.info) #model construction time too long
    model = getattr(models, opt.model)(train_data.info)
    # end_time = time()
    # model = LEnsemble(train_data.info)
    # model = compElemDPP(train_data.info)

    # print('time elapsed for model building: %d' %(end_time-start_time))
    outputs = []
    for i in tqdm(train_dl,leave=True):
        out=model(i)
        outputs.append(out)
        # break
    outputs = torch.cat(outputs)
    print(torch.exp(outputs).sum()) #check if sum is 1
    # print(outputs.sum())


    pass
    
    
    
