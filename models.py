from turtle import forward
import torch
import torch.nn as nn


#see how to pass children's result to parent
class sumNode(nn.Module):
    def __init__(self, parent, children, leaf=False) -> None:
        super().__init__()
        num_child = len(children) if type(self.children) is not int else children
        self.W = nn.Parameter(torch.rand(num_child,1), requires_grad=True)
        self.parent = parent #single node 
        self.children = children #list
        self.val = None
        self.leaf=leaf
    def forward(self,x=None):
        """
        x should be in size num_child x 1
        """
        if not self.leaf:
            out = []
            for i in self.children:
                out.append(i(x))
            out = torch.cat(out,dim=1)
            
        else:
            #'leaf sum node'
            
            out=[]
            l = len(self.children)
            
            for i in range(l):
                temp = (x^True)[:,:l]
                temp[:,i]=1

                out.append(x[:,i]*torch.prod(temp,dim=1))

            out = torch.stack(out,dim=1)
        out = torch.matmul(out.to(torch.float),self.W)
        
        return out 

class prodNode(nn.Module):
    def __init__(self, parent, children) -> None:
        super().__init__()
        self.parent = parent #single node 
        self.children = children #list or index to input 
        self.val = None
    def forward(self,x=None):
        
        out = []
        for i in self.children:
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


class LEsemble(nn.Module):
    def __init__(self, data_info) -> None:
        super().__init__()
        n = data_info['n']
        B = torch.randn(n, n)
        B_norm = torch.norm(B, dim=0)
        for i in range(0, n):
            B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)
        self.I = torch.eye(n)

        

    def forward(self,x): #output log likelihood
        #x.shape = B, n
        eps = 1e-8
        L = torch.matmul(self.B.transpose(0,1),self.B) + eps*self.I
        # L = self.L.unsqueeze(0).repeat(x.shape[0],1,1)

        # norm = torch.det(L+self.I) #results in inf?
        # still not sure how to enforce k constraints? 
        log_norm = torch.logdet(L+self.I)
        outputs = []
        for i in x:
            idx = (i==1)
            L_y = L[idx,:][:,idx]
            out = log_norm-torch.logdet(L_y) #negative log likelihood
            outputs.append(out)
        outputs = torch.stack(outputs)
        return outputs

        pass




if __name__=="__main__":
    from utils import *
    from torch.utils.data import DataLoader

    from time import time

    
    opt=parse_args()
    train_data=DatasetFromFile('nips')
    train_dl = DataLoader(train_data, batch_size=4)

    # start_time = time()
    # model=naivePC(train_data.info) #model construction time too long
    # end_time = time()
    model = LEsemble(train_data.info)

    # print('time elapsed for model building: %d' %(end_time-start_time))

    for i in train_dl:
        out=model(i)
        break

    print(out)


    pass
    
    
    
