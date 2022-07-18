from ast import parse
import numpy as np
from utils import *
from train import *
import copy
import torch
import torch.nn as nn
#random initializations


# cluster_a = np.array([0.1, 0.4, 0.3, 0.5, 0.25])
# cluster_b = np.array([0.37, 0.42, 0.15, 0.2, 0.7])

# # experiment_1 = np.array([True, False, False, True, True])

# data=DatasetFromFile('sanity_check')
# experiments = data.x.cpu().detach().numpy()
# print('import done')
def EM_logll(alphas, betas,X):

    W = (betas[None,:,:]**(X[:,:,None]))*((1-betas[None,:,:])**((1-X[:,:,None])))
    W = torch.prod(W,dim=1) #N,K
    P = W*alphas #alphas = 1,K
    P = P.sum(dim=1) #P.shape = N, P[i] likelihood of example i
    avg_ll = torch.log(P).mean()
    return avg_ll




if __name__=="__main__":
    torch.manual_seed(0)
    ##set up log directories
    opt = parse_args()
    opts = []
    
    time_stamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    rand_id='id'+str(randint(0,20))
    opt.output_folder = time_stamp +'_' +rand_id + "EM"
    K = opt.EM_cluster_num
    for i in range(K):
        #split opt
        temp_opt = copy.deepcopy(opt)
        temp_opt.output_dir = 'EM_cluster_%d'%i #each cluster has its own output directory
        process_opt(temp_opt)
        opts.append(temp_opt)

    #Load data
    data_pack = import_data(opt)
    train_data = data_pack[0]
    n = train_data.info['n']
    N = train_data.x.shape[0]
    
    #weights initializations: alpha, beta
    alphas = nn.functional.softmax(torch.randn(opt.EM_cluster_num),dim=0)
    betas = torch.rand(n,K) #initilize bernoulli prob for each variable in each cluster
    EM_training_log_path = 'log/'+opt.output_folder+'/EM_log.txt'
    print('EM steps starts')
    for step in tqdm(range(opt.EM_steps),leave=True):
        #E-step
        #W[i,k] probability that train_data.x[i] belongs to cluster k, W.shape = N, K
        W = (betas[None,:,:]**(train_data.x[:,:,None]))*((1-betas[None,:,:])**((1-train_data.x[:,:,None])))
        W = torch.prod(W,dim=1)
        # W = torch.prod((betas[None,:,:]**(train_data.x[:,:,None])),dim=1)*torch.prod(((1-betas[None,:,:])**((1-train_data.x[:,:,None]))),dim=1)
        #normalize W
        W = W/W.sum(dim=1,keepdim=True)
        N_k = W.sum(dim=0,keepdim=True)
        alphas = N_k/N #update alphas

        #M-step
        betas = ((W[:,:,None]*train_data.x[:,None,:]).sum(dim=0)).transpose(1,0)/N_k #betas.shape = n, K

        #compute log likelihood
        ll = EM_logll(alphas, betas,train_data.x)
        with open(EM_training_log_path, 'a+') as f:
            f.write('EM avg ll: %.4f \n'%(ll))
        print('done')


    #Train each clusters
    


