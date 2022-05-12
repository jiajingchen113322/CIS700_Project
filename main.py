import numpy as np
import torch
from dataloader import gestsets

import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
import argparse
from util import get_acc
import yaml
# ============== Get Configuration =================
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='dim_16_attention')
    cfg.add_argument('--epochs',default=20)
    cfg.add_argument('--train',action='store_true',default=True)
    cfg.add_argument('--data_path',default='babi_data\processed_2/train/15_graphs.txt')
    cfg.add_argument('--batch_size',default=15)
    cfg.add_argument('--lr',default=0.01)
    cfg.add_argument('--device',default='cpu')
    cfg.add_argument('--opt',default='adam'),
    cfg.add_argument('--attention',default=True,action='store_true')
    
    # ==== model config =====
    cfg.add_argument('--state_dim',default=16)
    cfg.add_argument('--annotation_dim',default=1)
    cfg.add_argument('--edge_type',default=2)
    cfg.add_argument('--n_node',default=8)
    cfg.add_argument('--n_step',default=5)
    # =======================
    
    # ==== prject path ====
    cfg.add_argument('--proj_path',default='D:\Courses/2022 Spring\Theorem Proving\Final_project\My_Project')
    # =====================

    return cfg.parse_args()

cfg=get_arg()

if cfg.attention:
    from model_attention import GGNN
else:
    from model import GGNN

def main():
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
  
    cuda=0
    model=GGNN(state_dim=cfg.state_dim,annotation_dim=cfg.annotation_dim,n_edge_types=cfg.edge_type,n_node=cfg.n_node,n_steps=cfg.n_step)
    train_loader,valid_loader=gestsets(root=cfg.data_path,batch_size=cfg.batch_size)
    
    if cfg.train:
        train_model(model,train_loader,valid_loader,cfg.exp_name,cuda)
    
    else:
        # pth_folder=os.path.join('./pth_file',exp_name)
        # test(model,test_loader,pth_folder=pth_folder)
        print('not implemented yet')
        
        
def test(model,data_loader,pth_folder):
    print('no need for test here')



def train_model(model,train_loader,valid_loader,exp_name,cuda_n):
    assert torch.cuda.is_available()
    epoch_acc=[]
    model=model.to(cfg.device)


    initial_epoch=0
    training_epoch=cfg.epochs

    loss_func=nn.CrossEntropyLoss()
    if cfg.opt=='SGD':
        optimizer=torch.optim.SGD(model.parameters(),cfg.lr)
    else:
        optimizer=torch.optim.Adam(model.parameters(),cfg.lr)
    
    lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,training_epoch,30),gamma=0.7)


    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
     
        epsum=run_one_epoch(model,iteration,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        
        epoch_acc.append(mean_acc)
        
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['losses'])
        return summary



    #build tensorboard
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)

    #build folder for pth_file
    exp_path=os.path.join(cfg.proj_path,'Experiment',exp_name)
    if not os.path.exists(exp_path):
         os.mkdir(exp_path)
    
    pth_path=os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    
    tensorboard_path=os.path.join(exp_path,'TB')
    if not os.path.exists(tensorboard_path):
         os.mkdir(tensorboard_path)
    
    # ==== save configuration file =====
    cfg_dict=vars(cfg)
    yaml_file=os.path.join(exp_path,'config.yaml')
    with open(yaml_file,'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)
  
    
    tensorboard=SummaryWriter(log_dir=tensorboard_path)
    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary=eval_one_epoch()
        summary={**train_summary,**valid_summary}
        lr_schedule.step()
        #save checkpoint
        if np.max(epoch_acc)==epoch_acc[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict()}


            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    


def run_one_epoch(model,tqdm_iter,mode,loss_func=None,optimizer=None,loss_interval=10):
    if mode=='train':
        model.train()
    else:
        model.eval()
        param_grads=[]
        for param in model.parameters():
            param_grads+=[param.requires_grad]
            param.requires_grad=False
    
    summary={"losses":[],"acc":[]}
    device=next(model.parameters()).device

    for i,(am_cpu,anotation_cpu,target_cpu) in enumerate(tqdm_iter):
        # ==== get input ====
        padding = torch.zeros(len(anotation_cpu), cfg.n_node, cfg.state_dim - cfg.annotation_dim)
        init_input = torch.cat((anotation_cpu, padding), 2).to(cfg.device)
        adj_matrix = am_cpu.to(cfg.device)
        annotation = anotation_cpu.to(cfg.device)
        target = target_cpu.to(cfg.device)
        # ===================
        
        if mode=='train':
            optimizer.zero_grad()
        
        logits=model(init_input.float(),annotation.float(),adj_matrix.float()) # (bs,node_num)
        if loss_func is not None:
            re_logit=logits.reshape(-1,logits.shape[-1])

            #### here is the loss #####
            loss=loss_func(re_logit,target.view(-1))
            summary['losses']+=[loss.item()]
        
        if mode=='train':
            loss.backward(retain_graph=True)
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: {:.3f}".format(np.mean(summary['losses'])))

        else:
            log=logits.cpu().detach().numpy()
            lab=target.detach().cpu().numpy()
            
            mean_acc=get_acc(log,lab)
            summary['acc'].append(mean_acc)

    
            if i%loss_interval==0:
                tqdm_iter.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))


    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value


    return summary


if __name__=='__main__':
    main()
