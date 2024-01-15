#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import sys
import os
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist

import json

from data_utils import get_UnetDataset, collate_fn_speech_enhancement
from complexunet import ComplexU_Net, ComplexU_Net_Attention, ComplexU_Net_Channel_Attention
from loss_functions import Unet2Loss


def main(params):
    test_array_filepath = params['data']['test_array']
    train_array_filepath = params['data']['train_array']
    nch = 3
    checkpoint_interval = params['data']['checkpoint_interval']
    model_checkpoint_dir = params['data']['model_dir']
    dirpath = params['data']['dirpath']

    test_dataset = get_UnetDataset(dirpath, test_array_filepath)

    train_dataset = get_UnetDataset(dirpath, train_array_filepath)


    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_speech_enhancement, drop_last=True, num_workers=4)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_speech_enhancement, drop_last=True, num_workers=4)

    dataloader = {'train':train_dataloader, 'test':test_dataloader}

    if model_checkpoint_dir != '' and os.path.exists(model_checkpoint_dir) == False:
        os.mkdir(model_checkpoint_dir)

    learning_rate = params['model']['learning_rate']
    weight_decay =  params['model']['weight_decay']
    start_epoch =  params['model']['start_epoch']
    num_epochs =  params['model']['num_epochs']
    checkpoint_model_path = params['data']['checkpoint_model_path']
    checkpoint_optimizer_path = params['data']['checkpoint_optimizer_path']
    use_cuda = bool(params['model']['use_cuda'])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ComplexU_Net(inchannel=4)

    model = model.double()
    #print(model)
    criterion = Unet2Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
    
    if torch.cuda.is_available():
        
        import GPUtil
        import sys
        gpu = GPUtil.getAvailable(order = 'memory')[0]

        cuda_gpu_device_id = 'cuda:%d'%gpu
        device = torch.device(cuda_gpu_device_id)
        
        gpu = GPUtil.getAvailable(order = 'memory')[0]

        cuda_gpu_device_id = 'cuda:%d'%gpu
        print('model running on cuda device', cuda_gpu_device_id)
        device = torch.device(cuda_gpu_device_id)
        model = model.to(device)
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
        
    if checkpoint_model_path != '' and checkpoint_optimizer_path != '':
        model.load_state_dict(torch.load(checkpoint_model_path, map_location=device))
        optimizer.load_state_dict(torch.load(checkpoint_optimizer_path, map_location=device))
        print('model, optimizer loaded .....')
        

    print('Training started .......')

    loss_history = {"train": [], "test": []}
    for epoch in range(start_epoch+1, num_epochs):
        for phase in ['train']:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i, data in enumerate(dataloader[phase]):
                y_stft, s_stft, mask, beam_stft, y, s, n = data
                y_stft = y_stft.unsqueeze(0)
                s_stft = s_stft.unsqueeze(0)
                beam_stft = beam_stft.unsqueeze(0).unsqueeze(0).permute(0,2,3,1,4) 
                mask = mask.unsqueeze(0)
                
                #print(y_stft.shape, beam_stft.shape)
                
                input_stft = torch.cat((y_stft, beam_stft.double()), dim=3)
                
                #print(y_stft.shape, input_stft.shape, s_stft.shape, mask.shape, beam_stft.shape)

                if use_cuda:
                    y_stft, input_stft, s_stft, mask, y, s, n = y_stft.to(device), input_stft.to(device), s_stft.to(device), mask.to(device), y.to(device), s.to(device), n.to(device)

                #print(input_stft.shape, mask.shape)

                pred_mask = model(input_stft.permute(0,3,2,1,4).double())
                #print(pred_mask.shape)#y_stft, s_stft, mask, pred_mask
                loss = criterion(y_stft, s_stft, mask, pred_mask, y, s, n, device)
                optimizer.zero_grad()

                if phase == 'train':
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                running_loss += loss.item()

                #print('loss', loss)

            loss_history[phase].append(running_loss / len(dataloader[phase]))

                
        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(model_checkpoint_dir,'unet_model_epoch_'+str(epoch+1)+'.pt'))
            torch.save(optimizer.state_dict(), os.path.join(model_checkpoint_dir,'unet_optimizer_epoch_'+str(epoch+1)+'.pt'))

            plt.plot(loss_history["train"], linewidth=2, label="Train loss")
            plt.plot(loss_history["test"], linewidth=2, label="Test loss")
            plt.legend(prop={"size": 16})
            plt.savefig('train_test_loss_plot.png')	


if __name__ == "__main__":
    config_filepath = sys.argv[1] # argument 1 config file
    
    if os.path.exists(config_filepath) == False:
        print('Please check config filepath', config_filepath)
    else:
        with open(config_filepath, 'r') as f:
            params = json.load(f)
    
        main(params)
