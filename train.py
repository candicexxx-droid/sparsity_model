from utils import *
import models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def nll(y):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    ll = -torch.sum(y)
    return ll
def avg_ll(model, dataset_loader):
    """
    adapted from https://github.com/joshuacnf/Probabilistic-Generating-Circuits/blob/18700951ad18759e95ca85430da66042931b6c8b/pgc/train.py#L163
    """
    lls = []
    dataset_len = 0
    for x_batch in dataset_loader:
        # x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch.shape[0]
    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len
    return avg_ll

def main(opt):
    train_data, valid_data, test_data= DatasetFromFile(opt.data),DatasetFromFile(opt.data, 'valid'),DatasetFromFile(opt.data,'test')
    train_dl, valid_dl, test_dl = DataLoader(train_data, batch_size=opt.batch_size),DataLoader(valid_data, batch_size=opt.batch_size),DataLoader(test_data, batch_size=opt.batch_size)
    model = getattr(models, opt.model)(train_data.info)

    optimizer = optim.SGD(model.parameters(),opt.lr,opt.momentum,opt.weight_d) if opt.optimizer =='SGD' else optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_d)
    
    tb_writer = SummaryWriter(log_dir=opt.output_dir)
    
    hyperparam=vars(opt)
    hyperparam_file = open(opt.output_dir+"/hyperparam.json","w")

    for epoch in range(1, opt.epoch + 1):

        model.train()
        losses=[]
        for x_batch in tqdm(train_dl,leave=True):
            # x_batch = x_batch.to(device)
            y_batch = model(x_batch)
            loss = nll(y_batch)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        #logging performance/testing  
        
        avg_loss = sum(losses)/len(losses) #avg loss for each epoch
        print('Dataset {}; Epoch {}, Batch avg Loss: {}'.format(opt.data, epoch, avg_loss))
        tb_writer.add_scalar("%s/avg_loss"%"train", avg_loss, epoch)
        # compute likelihood on train, valid and test
        train_ll = avg_ll(model, train_dl)
        valid_ll = avg_ll(model, valid_dl)
        # test_ll = avg_ll(model, test_dl)

        tb_writer.add_scalar("%s/avg_ll"%"train", train_ll, epoch)
        tb_writer.add_scalar("%s/avg_ll"%"valid", train_ll, epoch)
        # tb_writer.add_scalar("%s/avg_ll"%"test", train_ll, epoch)
    
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
    main(opt)
    print('done')
