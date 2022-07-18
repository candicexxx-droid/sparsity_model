from utils import *
import models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import json

def avg_ll(model, dataset_loader,device):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    lls = []
    dataset_len = 0
    model.eval()
    # is_multi = isinstance(model,models.multi_arrayPC)
    for x_batch in tqdm(dataset_loader,leave=True):
        # if is_multi:
        #     x_batch = [i.to(device) for i in x_batch]
        # else:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        # if is_multi:
        #     dataset_len += x_batch[0].shape[0]
        # else:
        dataset_len += x_batch.shape[0]
    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len
    return avg_ll

def import_data(opt):

    return DatasetFromFile(opt.data),DatasetFromFile(opt.data,'valid'),DatasetFromFile(opt.data,'test')

def main(opt, data_pack):


    device_name="cuda:%d"%opt.cuda if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_data, valid_data, test_data= data_pack
    train_dl, valid_dl, test_dl = DataLoader(train_data, batch_size=opt.batch_size),DataLoader(valid_data, batch_size=opt.batch_size),DataLoader(test_data, batch_size=opt.batch_size)
    if "sum" in opt.model:
        model = getattr(models, opt.model)(train_data.info, opt.group_num)
    else:
        model = getattr(models, opt.model)(train_data.info)
    model = model.to(device)
    is_multi = isinstance(model,models.multi_arrayPC) or isinstance(model,models.sum_arrayPCs)
    if is_multi:
        param = []
        for i in model.array_PCs:
            param += list(i.parameters())
    else:
        param = list(model.parameters())
    num_param = sum([torch.prod(torch.tensor(i.shape)).item() for i in param])
    print('number of param: %d'%num_param)
    optimizer = optim.SGD(param,opt.lr,opt.momentum,opt.weight_d) if opt.optimizer =='SGD' else optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_d)
    
    tb_writer = SummaryWriter(log_dir=opt.output_dir)
    
    hyperparam=vars(opt)
    hyperparam_file = open(opt.output_dir+"/hyperparam.json","w")
    json.dump(hyperparam, hyperparam_file)
    hyperparam_file.close()

    log_file = opt.output_dir+"/log.txt"
    with open(log_file, 'a+') as f:
        f.write('number of param: %d \n'%(num_param))
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1, opt.epoch + 1):
            
            model.train()
            losses=[]
            num_items = []
            for x_batch in tqdm(train_dl,leave=True):
                
                # if is_multi and isinstance(model,models.multi_arrayPC):
                #     x_batch = [i.to(device) for i in x_batch]
                # else:
                x_batch = x_batch.to(device)
                y_batch = model(x_batch)
                loss = nll(y_batch)
                losses.append(loss)
                # if is_multi and isinstance(model,models.multi_arrayPC):
                #     num_items.append(x_batch[0].shape[0])
                # else:
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
            train_ll = -avg_loss
            valid_ll = avg_ll(model, valid_dl,device)
            # test_ll = avg_ll(model, test_dl)

            tb_writer.add_scalar("%s/avg_ll"%"train", -avg_loss, epoch)
            tb_writer.add_scalar("%s/avg_ll"%"valid", valid_ll, epoch)
            
            with open(log_file, 'a+') as f:
                f.write('%s Epoch: %d train: %.5f validation: %.5f \n'%(opt.model, epoch, train_ll, valid_ll))
            torch.save({
            'epoch': epoch,
            
            'model':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            
            }, opt.output_dir+"/end_chpt.pt")



        test_ll = avg_ll(model, test_dl,device)
        with open(log_file, 'a+') as f:
            f.write('End test: %.5f \n'%test_ll)
        tb_writer.add_scalar("%s/avg_ll"%"test", test_ll, epoch)
    
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
    pack = import_data(opt)
    main(opt, pack)
    print('done')
