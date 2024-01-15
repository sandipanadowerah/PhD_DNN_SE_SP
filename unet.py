import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

class ConvBlock_Self_attention(nn.Module):
    def __init__(self, inchannel, outchannel, embed_dim, num_heads = 8, maxpool = True):
        super(ConvBlock_Self_attention, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = 1
        padding = 1
        num_heads = num_heads
        embed_dim = embed_dim
        self.conv0 = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size, stride, padding)
        
        self.multihead_attention = Multihead_Attention(outchannel, embed_dim, num_heads)
        
        self.batch_norm = nn.BatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = nn.MaxPool2d(maxpool_size)
        self.relu = nn.ReLU()

        
    def forward(self, x): 
        x = self.relu(self.conv0(x))
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = self.relu(self.maxpool_layer(x))
        else:
            x = self.relu(x)
        
        #print(x.shape)
        x = self.multihead_attention(x)
        
        return x, x
    
    

class ConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel, maxpool = True):
        super(ConvBlock, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = 1
        padding = 1
        
        self.conv0 = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = nn.MaxPool2d(maxpool_size)
        self.relu = nn.ReLU()

        
    def forward(self, x): 
        x = self.relu(self.conv0(x))
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = self.relu(self.maxpool_layer(x))
        else:
            x = self.relu(x)
        return x, x

class ConvBlock_channel_attention(nn.Module):
    def __init__(self, inchannel, outchannel, embed_dim, num_heads = 8, maxpool = True):
        super(ConvBlock_channel_attention, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = 1
        padding = 1
        num_heads = num_heads
        embed_dim = embed_dim
        self.conv0 = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size, stride, padding)
        
        self.multihead_attention = Multihead_Attention(embed_dim, outchannel, num_heads)
        
        self.batch_norm = nn.BatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = nn.MaxPool2d(maxpool_size)
        self.relu = nn.ReLU()

        
    def forward(self, x): 
        #print(x.shape)
        x = self.relu(self.conv0(x))
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = self.relu(self.maxpool_layer(x))
        else:
            x = self.relu(x)
        
        #print(x.shape)  # torch.Size([1, 16, 64, 128])
        x = x.permute(0,3,2,1)
        x = self.multihead_attention(x)
        x = x.permute(0,3,2,1)
        #print(x.shape)
        
        return x, x

    

class UpConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel, up=True):
        super(UpConvBlock, self).__init__()
        
        self.up = up
        kernel_size = 3
        stride = 1
        padding = 1
        if self.up == True:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        self.deconv = nn.ConvTranspose2d(inchannel, outchannel, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

        
    def forward(self, x, r):
        if self.up == True:
            x = self.upsample(x)
        x = torch.cat([x,r], dim=1)
        x = self.deconv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        return x

class Multihead_Attention(nn.Module):
    def __init__(self, in_ch, embed_dim, num_heads):
        super(Multihead_Attention, self).__init__()
        
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.split_size = embed_dim // num_heads
        
        self.conv_query =  nn.Conv2d(in_ch, in_ch,1,1)
        self.conv_key   =  nn.Conv2d(in_ch, in_ch,1,1)
        self.conv_value =  nn.Conv2d(in_ch, in_ch,1,1)

        
        
    def forward(self, x): 
        
        q = self.conv_query(x)
        k = self.conv_key(x)
        v = self.conv_value(x)
        #print(q.shape, k.shape, v.shape)
        querys = torch.stack(torch.split(q, self.split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(k, self.split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(v, self.split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        #print(querys.shape, keys.shape, values.shape)
        scores = torch.matmul(querys, keys.transpose(3, 4))  # [h, N, T_q, T_k]
        scores = scores / (self.embed_dim ** 0.5)
        #print(scores.shape)
        scores = F.softmax(scores, dim=4) # need to find right dim #TODO
        
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        return out





class U_Net_Attention(nn.Module): #TODO
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(U_Net_Attention, self).__init__()
        
        self.inchannel = inchannel #3
        
        self.conv1 = ConvBlock_Self_attention(inchannel, filters[0], filters[3], num_heads = 8)
        self.conv2 = ConvBlock_Self_attention(filters[0], filters[1], filters[2], num_heads = 4)
        self.conv3 = ConvBlock_Self_attention(filters[1], filters[2], filters[1], num_heads = 2)
        self.conv4 = ConvBlock_Self_attention(filters[2], filters[3], filters[0], num_heads = 1)
        self.conv5 = ConvBlock_Self_attention(filters[3], filters[4], filters[0], num_heads = 1, maxpool = False)
        
        self.deconv4 = UpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = UpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = UpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = UpConvBlock(filters[1]+filters[0], filters[0])
        
        
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
        self.conv00 = nn.ConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 = nn.Linear(256, output_dim)
        self.conv01 = nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, x): # input : 1 x 3 x 21 x 257
        
        # encoder
        x, z1 = self.conv1(x)
        #print('conv1', x.shape)
        x, z2 = self.conv2(x)
        #print('conv2', x.shape)
        x, z3 = self.conv3(x)
        #print('conv3', x.shape)
        x, z4 = self.conv4(x)
        #print('conv4', x.shape)
        
        # latent space
        x, _ = self.conv5(x)
        
        #print('conv5', x.shape, z4.shape) # decoder
        x = self.deconv4(x, z4)
        x = self.deconv3(x, z3)
        x = self.deconv2(x, z2)
        x = self.deconv1(x, z1)
        
        # postnet
        x = self.up1(x)
        x = self.conv00(x)
        x = self.sigmoid(self.fc0(x))
        x = self.conv01(x)
        
        return x

class U_Net(nn.Module):
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(U_Net, self).__init__()
        
        self.inchannel = inchannel #3
        
        self.conv1 = ConvBlock(inchannel, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4], maxpool = False)
        
        self.deconv4 = UpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = UpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = UpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = UpConvBlock(filters[1]+filters[0], filters[0])
        
        
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
        self.conv00 = nn.ConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 = nn.Linear(256, output_dim)
        self.conv01 = nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, x): # input : 1 x 3 x 21 x 257
        
        # encoder
        x, z1 = self.conv1(x)
        #print('conv1', x.shape)
        x, z2 = self.conv2(x)
        #print('conv2', x.shape)
        x, z3 = self.conv3(x)
        #print('conv3', x.shape)
        x, z4 = self.conv4(x)
        #print('conv4', x.shape)
        
        # latent space
        x, _ = self.conv5(x)
        
        # decoder
        x = self.deconv4(x, z4)
        x = self.deconv3(x, z3)
        x = self.deconv2(x, z2)
        x = self.deconv1(x, z1)
        
        # postnet
        x = self.up1(x)
        x = self.conv00(x)
        x = self.sigmoid(self.fc0(x))
        x = self.conv01(x)
        
        return x


class U_Net_Channel_Attention(nn.Module):#TODO
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(U_Net_Channel_Attention, self).__init__()
        
        self.inchannel = inchannel #3
        
        self.conv1 = ConvBlock_channel_attention(inchannel, filters[0], filters[3], num_heads = 8)
        self.conv2 = ConvBlock_channel_attention(filters[0], filters[1], filters[2], num_heads = 8)
        self.conv3 = ConvBlock_channel_attention(filters[1], filters[2], filters[1], num_heads = 8)
        self.conv4 = ConvBlock(filters[2], filters[3])#, filters[0], num_heads = 2)
        self.conv5 = ConvBlock(filters[3], filters[4], maxpool=False)#, filters[0], num_heads = 1, maxpool = False)
        
        self.deconv4 = UpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = UpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = UpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = UpConvBlock(filters[1]+filters[0], filters[0])
        
        
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
        self.conv00 = nn.ConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 = nn.Linear(256, output_dim)
        self.conv01 = nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, x): # input : 1 x 3 x 21 x 257
        
        # encoder
        #print(x.shape)
        x, z1 = self.conv1(x)
        #print('conv1', x.shape)
        x, z2 = self.conv2(x)
        #print('conv2', x.shape)
        x, z3 = self.conv3(x)
        #print('conv3', x.shape)
        x, z4 = self.conv4(x)
        #print('conv4', x.shape)

        #print('x.shape, z4.shape', x.shape, z4.shape)
        
        # latent space
        x, _ = self.conv5(x)

        #print('x.shape conv5', x.shape)
        
        #print('x, z4 deconv4', x.shape, z4.shape)# decoder
        x = self.deconv4(x, z4)
        x = self.deconv3(x, z3)
        x = self.deconv2(x, z2)
        x = self.deconv1(x, z1)
        
        # postnet
        x = self.up1(x)
        x = self.conv00(x)
        x = self.sigmoid(self.fc0(x))
        x = self.conv01(x)
        
        return x

    



#    model = U_Net() 
#    #print(model)
#    input_ = torch.rand([1, 3, 700, 257])
#    output_ = torch.rand([1, 1, 1, 257])
#    pred_ = model(input_)
#    print(pred_.shape)
    
#    torch.save(model.state_dict(), 'Unet_self_attention_model.pt')
