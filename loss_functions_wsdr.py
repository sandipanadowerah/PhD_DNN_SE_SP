import torch
import torch.nn as nn
from istft import *

def l1_loss(clean_hat: torch.Tensor, clean: torch.Tensor):
        return torch.abs(clean_hat - clean).mean()

def wsdr_loss(clean_hat: torch.Tensor, clean: torch.Tensor, noise: torch.Tensor, eps: float = 1e-5):
    # calc norm
    clean_norm = clean.norm(dim=1)
    clean_hat_norm = clean_hat.norm(dim=1)
    minus_c_norm = (noise - clean).norm(dim=1)
    minus_ch_norm = (noise - clean_hat).norm(dim=1)

    # calc alpha
    alpha = clean_norm ** 2 / (clean_norm ** 2 + minus_c_norm ** 2 + eps)

    # calc loss
    loss_left = - alpha * (clean * clean_hat).sum(dim=1) / (clean_norm * clean_hat_norm + eps)
    loss_right = - (1 - alpha) * ((noise - clean) * (noise - clean_hat)).sum(dim=1) / (minus_c_norm * minus_ch_norm + eps)
    loss = (loss_left + loss_right).mean() 
    
    return loss

def power_loss(clean_hat: torch.Tensor, clean: torch.Tensor):
        # Power Loss on "ClariNet" paper
        # https://arxiv.org/pdf/1807.07281.pdf
        B = clean_hat.size(1)
        return (clean_hat - clean).norm() / B

def stft_to_wav(o_stft, len_):
    n_fft = 512
    hop_length = 256
    complex_ = torch.zeros_like(o_stft)
    real_ = o_stft.clone()
    _stft = torch.cat([real_, complex_], dim=0).unsqueeze(0).permute(0,2,3,1)
    wav = istft(_stft, hop_length, length=len_)
    
    return wav


class Unet2Loss(nn.Module):
    def __init__(self):
        super(Unet2Loss, self).__init__()

    def forward(self, y_stft, s_stft, mask, pred_mask, s_dry, y):
        
        #print(y_stft.shape, s_stft.shape, mask.shape, pred_mask.shape)

        #print(y_stft[:,0,:,:].shape, pred_mask[:,0,:,:].shape) #, pred_mask.shape, mask[:,:1,:,:].shape)
        
        pred_mask = pred_mask.permute(0,1,3,2)
        y_stft = y_stft.permute(0,1,3,2)
        o_stft = pred_mask[:,0,:,:]*y_stft[:,0,:,:]
        y_o = stft_to_wav(o_stft, len_=s_dry.shape[0])

        pred_mask = pred_mask.permute(0,1,3,2)
        #print('shapes, clean, noise, output', s_dry.shape, y.shape, y_o.shape)
        
        #stft_loss = nn.MSELoss()(o_stft, s_stft[:,:,:,0]) 
        #mask_loss = nn.MSELoss()(pred_mask, mask[:,:1,:,:])
        
        l1_l = l1_loss(y_o.T, s_dry.T)
        power_l = power_loss(y_o.T, s_dry.T)
        wsdr_l = wsdr_loss(y_o.T, s_dry.T, y.T)
        #print(mask_loss.item(),stft_loss.item(),l1_l.item(),power_l.item(),wsdr_l.item(), 'mask_mse, stft_mse l1_wav, power_l, wsdr')        
        unet_loss =  l1_l+ power_l + wsdr_l 

        return unet_loss
