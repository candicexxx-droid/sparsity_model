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
def EM_logll(alphas, betas,X, return_P=False):

    W = (betas[None,:,:]**(X[:,:,None]))*((1-betas[None,:,:])**((1-X[:,:,None])))
    W = torch.prod(W,dim=1) #N,K
    P = W*alphas #alphas = 1,K
    if return_P:
        group_assign = torch.argmax(P,dim=1)
        return group_assign
    P = P.sum(dim=1) #P.shape = N, P[i] likelihood of example i
    avg_ll = torch.log(P).mean()
    return avg_ll

def split_data(opt,data, group_assign, info):
    train_datasets = []
    for i in range(opt.EM_cluster_num):
        idx = (group_assign == i)
        x = data.x[idx]
        temp = DatasetFromFile(opt.data,'',x)
        temp.info = info
        train_datasets.append(temp)
    return train_datasets

def EM_eval(datasets,opt, sub_models,device):
    lls_list = torch.Tensor(0)
    total_data_num = 0
    for i in range(opt.EM_cluster_num):
        data = datasets[i]
        dl = DataLoader(data, batch_size=opt.batch_size)
        model = sub_models[i]
        lls,dataset_len = avg_ll(model,dl,device,True)
        lls_list = torch.cat((lls_list,lls))
        total_data_num+=dataset_len
    
    ll = torch.sum(lls_list).item() / total_data_num
    return ll




if __name__=="__main__":
    torch.manual_seed(0)
    ##set up log directories
    opt = parse_args()
    opts = []
    
    time_stamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    rand_id='id'+str(randint(0,20))
    opt.output_folder = time_stamp +'_' +rand_id + "EM-%s"%opt.data
    device_name="cuda:%d"%opt.cuda if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
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
    
    

    group_assign_train = EM_logll(alphas, betas,train_data.x, return_P=True)
    info = train_data.info
    
    
    # for i in range(opt.EM_cluster_num):
    #     idx = (group_assign == i)
    #     x = train_data.x[idx]
    #     temp = DatasetFromFile(opts[i].data,'train',x)
    #     temp.info = info
    #     train_datasets.append(temp)
    train_datasets = split_data(opt,train_data, group_assign_train, info)
    
    sub_models = []
    for i in range(opt.EM_cluster_num):
        opt = opts[i]
        sub_data_pack = (train_datasets[i],data_pack[1],data_pack[2])
        model = main(opt,sub_data_pack)
        sub_models.append(model)
    
    valid_data = data_pack[1]
    test_data = data_pack[2]
    
    group_assign_valid = EM_logll(alphas, betas,valid_data.x, return_P=True)
    valid_datasets = split_data(opt,valid_data, group_assign_valid, info)
    group_assign_test = EM_logll(alphas, betas,test_data.x, return_P=True)
    test_datasets = split_data(opt,test_data, group_assign_test, info)
    



    train_ll = EM_eval(train_datasets,opt,sub_models,device)
    valid_ll = EM_eval(valid_datasets,opt,sub_models,device)
    test_ll = EM_eval(test_datasets,opt,sub_models,device)
    # lls_list = torch.Tensor(0)
    # total_data_num = 0
    # for i in range(K):
    #     valid_data = valid_datasets[i]
    #     valid_dl = DataLoader(valid_data, batch_size=opt.batch_size)
    #     model = sub_models[i]
    #     lls,dataset_len = avg_ll(model,valid_dl,device,True)
    #     lls_list = torch.cat((lls_list,lls))
    #     total_data_num+=dataset_len
    
    # ll = torch.sum(lls_list).item() / total_data_num
    with open(EM_training_log_path, 'a+') as f:
        f.write('EM train avg ll using arrayPC mixture: %.4f \n'%(train_ll))
        f.write('EM validation avg ll using arrayPC mixture: %.4f \n'%(valid_ll))
        f.write('EM test avg ll using arrayPC mixture: %.4f \n'%(test_ll))
    print('done')


    #Train each clusters
    


