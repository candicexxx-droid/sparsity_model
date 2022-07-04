from turtle import forward
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
        n = data_info['n']
        k = data_info['k']
        B = torch.randn(n, data_info['k'])
        B_norm = torch.norm(B, dim=0)
        # for i in range(0, k):
        #     B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True) #n *k 
        self.I = torch.eye(n)

        

    def forward(self,x): #output log likelihood
        #x.shape = B, n
        eps = 1e-8 #soften the k-constraints
        L = torch.matmul(self.B,self.B.transpose(1,0)) 
        # + eps*self.I
        # L = self.L.unsqueeze(0).repeat(x.shape[0],1,1)

        # norm = torch.det(L+self.I) #results in inf?
        # still not sure how to enforce k constraints? 
        log_norm = torch.logdet(L+self.I)
        outputs = []
        for i in x:
            idx = (i==1)
            L_y = L[idx,:][:,idx]
            out = torch.logdet(L_y)-log_norm # log likelihood
            outputs.append(out)
        outputs = torch.stack(outputs)
        return outputs

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
        torch.manual_seed(10)
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
        # self.F = torch.zeros()
    def forward(self, x):
         #x.shape = B, n
        # F = torch.log(torch.zeros(x.shape[0], self.n, self.k))
        F = torch.zeros(x.shape[0], self.n, self.k)
        device = x.device
        F = F.to(device)
        # base case
        F[:,:,0] = 1
        F[:,0,1] = x[:,0] #x_1
        for i in range(0,self.n):
            for j in range (0, i+1):
                F[:,i,0] *= x[:,j]^True
        # print('base case done')
        # F[:,:,0]=torch.log(F[:,:,0])
        p_inf = torch.tensor(-float('inf'))
        # p_inf = -10**(10)
        F += self.episl
        F = torch.log(F)
        F[F.isinf()]= p_inf
        W = nn.functional.softmax(self.W,dim=2)
        W_full = W * self.W_adjust1 + self.W_adjust2 + self.episl
        # 
        # W_full = W
        W_full = torch.log(W_full)   
        for i in range(1, self.n):
            W = W_full[i-1]#shape self.k-1, 2

            # if i< self.k-1:
            #     W_adjust1 = torch.zeros(W.shape)
            #     W_adjust1[:i] = 1
            #     W_adjust2 = torch.zeros(W.shape)
            #     W_adjust2[i][0] = 1
            #     W = W * W_adjust1 + W_adjust2
            
            # W = 
            prior = torch.stack([F[:,i-1,:self.k-1].clone(), F[:,i-1,1:].clone()],dim=2) #shape B, self.k-1, 2
            
            log_x_part = torch.log(x[:,i].unsqueeze(1)+self.episl)
            log_x_part[log_x_part.isinf()]= p_inf

            log_x_bar_part = torch.log((x[:,i]^True).unsqueeze(1)+self.episl)
            log_x_bar_part[log_x_bar_part.isinf()]= p_inf

            prior[:,:,0] += log_x_part
            prior[:,:,1] += log_x_bar_part
            F[:,i,1:] = torch.logsumexp(W+prior,dim=2)
            # print(' ')
            # F[:,i,1:] += 
            # print('done')
            # #x[:,i].shape = B,1
            pass 

        endW = torch.log(nn.functional.softmax(self.endW,dim=1))
        # out = torch.matmul(F[:,-1,:],)
        # non_inf = ~F[:,-1,:].isinf()
        # endW = endW.repeat(x.shape[0],1)[non_inf]
        # # out = torch.logsumexp(endW+F[:,-1,:][non_inf],dim=0)
        # out = endW+F[:,-1,:][non_inf]
        # out = torch.log(out)

        out = torch.max(endW+F[:,-1,:],dim = -1)[0]



        return out

class arrayPC_naive(arrayPC):
    def __init__(self, data_info)-> None:
        super().__init__(data_info)
        # self.n, self.k = data_info['n'],(data_info['k']+1)
        # self.W = nn.Parameter(torch.randn(self.n-1,self.k-1,2), requires_grad=True)
        # self.endW = self.endW.transpose(1,0)
        # self.F = torch.zeros()
        self.endW.data = self.endW.data.transpose(1,0)
    def forward(self, x):
         #x.shape = B, n
        F = torch.zeros(x.shape[0], self.n, self.k)
        # base case
        F[:,:,0] = 1 
        F[:,0,1] = x[:,0] #x_1
        
        W = nn.functional.softmax(self.W,dim=2)
        W_full = W * self.W_adjust1 + self.W_adjust2

        # W_full = 
        # W_full = 
        # 


        for i in range(0,self.n):
            for j in range (0, i+1):
                F[:,i,0] *= x[:,j]^True
        # print('base case done')
        for i in range(1, self.n):
            W = W_full[i-1]
            
            # if i< self.k-1:
                
                
                 #shape self.k-1, 2
            F[:,i,1:] = x[:,i].unsqueeze(1)*W[:,0]*F[:,i-1,:self.k-1].clone() + (x[:,i]^True).unsqueeze(1)*W[:,1]*F[:,i-1,1:].clone() #
            
            # F[:,i,1:] += 
            # print('done')
            # #x[:,i].shape = B,1
            pass 

        
        endW = nn.functional.softmax(self.endW,dim=0)
        out = torch.matmul(F[:,-1,:],endW)
        out = torch.log(out)
        if (out > 1).sum()>0:
            print('pausze')
        return out

            





if __name__=="__main__":
    from utils import *
    from sanity_check_gen import *
    from torch.utils.data import DataLoader
    import models
    from time import time

    sanity_check_gen(20,4)
    opt=parse_args()
    train_data=DatasetFromFile('sanity_check')
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
    for i in train_dl:
        out=model(i)
        outputs.append(out)
        # break
    outputs = torch.cat(outputs)
    print(torch.exp(outputs).sum()) #check if sum is 1


    pass
    
    
    
