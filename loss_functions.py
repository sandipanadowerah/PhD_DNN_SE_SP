import torch
import torch.nn as nn

class Unet2Loss(nn.Module):
    def __init__(self):
        super(Unet2Loss, self).__init__()

    def forward(self, y_stft, s_stft, mask, pred_mask):
        
        #print(y_stft.shape, s_stft.shape, mask.shape, pred_mask.shape)

        #print(y_stft[:,0,:,:].shape, pred_mask[:,0,:,:].shape) #, pred_mask.shape, mask[:,:1,:,:].shape)
        
        pred_mask = pred_mask.permute(0,1,3,2)
        y_stft = y_stft.permute(0,1,3,2)
        o_stft = pred_mask[:,0,:,:]*y_stft[:,0,:,:]
        pred_mask = pred_mask.permute(0,1,3,2)

        #print('mse', o_stft.shape, s_stft[:,:,:,0].shape)
        
        stft_loss = nn.MSELoss()(o_stft, s_stft[:,:,:,0]) 
        mask_loss = nn.MSELoss()(pred_mask, mask[:,:1,:,:])

        #print(stft_loss, mask_loss)
        #print()
        #print()
       
        unet_loss =  mask_loss + stft_loss 

        #print(mask_loss, stft_loss)

        return unet_loss
