import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utility import *
from selex import *

def create_args_vm():
    args = Args()
    args.dataset = 'doxycol'
    args.k=1250
    args.n_componets=30
    args.timepoints = [1,2,3,4,5,6,7,8,9,10]
    args.niters = 5000
    args.lr = 3e-3
    args.num_samples = 200
    args.hidden_dim =  16
    args.n_hiddens =4
    args.activation =  'Tanh'
    args.gpu = 0
    args.input_dir = './all'
    args.save_dir = './doxy_out'
    args.seed = 1
    return args
    
    
if __name__ == '__main__':
    args=create_args_vm()

    torch.enable_grad()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:' + str(args.gpu)
                            if torch.cuda.is_available() else 'cpu')
    # load dataset
    integral_time = args.timepoints
    data_train,count_train=loaddata_selex(args,device)
    #data_train = loaddata(args,device)
    

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]


    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)
    func.apply(initialize_weights)


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            
            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,count_train,train_time,integral_time,sigma_now,options,device,itr)

            
            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())
            
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
            
            if itr % 500 == 0:
                ckpt_path = os.path.join(args.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                
            
            

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))


    
    
    
    