from turtle import forward
import torch
import torch.nn as nn


#see how to pass children's result to parent
class sumNode(nn.Module):
    def __init__(self, parent, children) -> None:
        super().__init__()
        num_child = len(children) if type(self.children) is not int else children
        self.W = nn.Parameter(torch.randn(num_child), requires_grad=True)
        self.parent = parent #single node 
        self.children = children #list
        self.val = None
    def forward(self,x=None):
        """
        x should be in size num_child x 1
        """
        if type(self.children) is not int:
            out = []
            for i in self.children:
                out.append(i(x))
            
        else:
            #'leaf sum node'
            c=x
            out=[]
            for i in range(self.children):
                temp = x^True
                temp[i]=1
                out.append(x[i]*torch.prod(temp))

        out = torch.as_tensor(out)
        out = torch.matmul(self.W, out)
        return out 

class prodNode(nn.Module):
    def __init__(self, parent, children) -> None:
        super().__init__()
        self.parent = parent #single node 
        self.children = children #list or index to input 
        self.val = None
    def forward(self,x=None):
        if type(self.children) is not int:
            out = []
            for i in self.children:
                out.append(i(x))
            out = torch.as_tensor(out)
        else:
            #'leaf product node'
            out = x[:self.children]
        
        out = torch.prod(out)
        return out

class inputNode(nn.Module):
    def __init__(self,data_idx) -> None:
        super().__init__()
        self.idx=data_idx
    
    def forward(self,x ):
        return x[self.idx]

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
            return prod_n
        if info['k']==1:
            sum_n=sumNode(None,info['n'])
            return sum_n
        info1={}
        info2={}


        info1['k']=info['k']
        info1['n']=info['n']-1 

        info2['k']=info['k']-1
        info2['n']=info['n']-1 



        return [prodNode(None, self._growPC(info1)), prodNode(None, [self._growPC(info2), inputNode(info['k']-1)])]

    def forward(self,x):

        return self.root(x)

if __name__=="__main__":
    from utils import *
    opt=parse_args()
    train_data=DatasetFromFile(opt.data)
    model=naivePC(train_data.info)
    out=model(train_data[0])



    pass
    
    
    
