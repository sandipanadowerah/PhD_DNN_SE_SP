import os
import torch
import torch.nn as nn

dirpath = '/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/sdowerah/data/dataset/train_dataset/DS_Mask'

for fname in os.listdir(dirpath):
    try:
        #print(data[0].shape, data[1].shape)
        fpath = os.path.join(dirpath, fname)
        data = torch.load(fpath)
        print(data[0].shape, data[1].shape)
    except:
        print('issue in ', fname)
    
    
    
    
