from utils import *
import models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import json


def nll(y):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    ll = -torch.sum(y)
    return ll
def avg_ll(model, dataset_loader,device):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    lls = []
    dataset_len = 0
    model.eval()
    for x_batch in dataset_loader:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch.shape[0]
    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len
    return avg_ll

def main(opt):


    device_name="cuda:%d"%opt.cuda if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_data, valid_data, test_data= DatasetFromFile(opt.data),DatasetFromFile(opt.data, 'valid'),DatasetFromFile(opt.data,'test')
    train_dl, valid_dl, test_dl = DataLoader(train_data, batch_size=opt.batch_size),DataLoader(valid_data, batch_size=opt.batch_size),DataLoader(test_data, batch_size=opt.batch_size)
    model = getattr(models, opt.model)(train_data.info)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),opt.lr,opt.momentum,opt.weight_d) if opt.optimizer =='SGD' else optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_d)
    
    tb_writer = SummaryWriter(log_dir=opt.output_dir)
    
    hyperparam=vars(opt)
    hyperparam_file = open(opt.output_dir+"/hyperparam.json","w")
    json.dump(hyperparam, hyperparam_file)
    hyperparam_file.close()


    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1, opt.epoch + 1):
            
            model.train()
            losses=[]
            num_items = []
            for x_batch in tqdm(train_dl,leave=True):

                x_batch = x_batch.to(device)
                y_batch = model(x_batch)
                loss = nll(y_batch)
                losses.append(loss)
                num_items.append(x_batch.shape[0])
                optimizer.zero_grad()
                # loss.backward(inputs=list(model.parameters()))
                loss.backward()
                optimizer.step()
                

            #logging performance/testing  
            
            # avg_loss = sum(losses)/len(losses) #avg loss for each epoch
            avg_loss = sum(losses)/sum(num_items) #avg loss for example
            print('Dataset {}; Epoch {}, avg Loss per example: {}'.format(opt.data, epoch, avg_loss))
            tb_writer.add_scalar("%s/avg_loss"%"train", avg_loss, epoch)
            # compute likelihood on train, valid and test
            train_ll = avg_ll(model, train_dl,device)
            valid_ll = avg_ll(model, valid_dl,device)
            # test_ll = avg_ll(model, test_dl)

            tb_writer.add_scalar("%s/avg_ll"%"train", train_ll, epoch)
            tb_writer.add_scalar("%s/avg_ll"%"valid", valid_ll, epoch)
        tb_writer.add_scalar("%s/avg_ll"%"test", train_ll, epoch)
    
    #save at the end of the epoch
    torch.save({
                'epoch': epoch,
                
                'model':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                
                }, opt.output_dir+"/end_chpt.pt")
    


if __name__=="__main__":
    opt = parse_args()
    process_opt(opt)
    if len(opt.sanity_check) !=0:
        from sanity_check_gen import *

        sanity_check_param = [int(i) for i in opt.sanity_check.split(',')]
        sanity_check_gen(sanity_check_param[0],sanity_check_param[1]) #generate sanity check data 
    main(opt)
    print('done')
