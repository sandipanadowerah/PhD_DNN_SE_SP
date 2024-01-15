import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d


class ComplexLinear(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_dim, out_dim):
        super().__init__()

        ## Model components
        self.linear_re = nn.Linear(in_dim, out_dim)
        self.linear_im = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.linear_re(x[:,:,:,:, 0]) #- self.linear_im(x[:,:,:,:, 1])
        imaginary = self.linear_im(x[:,:,:,:, 0])#self.linear_re(x[:,:,:,:, 1]) + self.linear_im(x[:,:,:,:, 0])
        output = torch.stack((real, imaginary), dim=-1)
        
        return output


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[:,:,:,:, 0]) - self.conv_im(x[:,:,:,:, 1])
        imaginary = self.conv_re(x[:,:,:,:, 1]) + self.conv_im(x[:,:,:,:, 0])
        output = torch.stack((real, imaginary), dim=-1)
        
        return output
    
class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[:,:,:,:, 0]) - self.tconv_im(x[:,:,:,:, 1])
        imaginary = self.tconv_re(x[:,:,:,:, 1]) + self.tconv_im(x[:,:,:,:, 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output
    
    
class complex_max_pool2d(nn.Module):
    def __init__(self,  kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        super().__init__()

        ## Model components
        self.max_pool_re   = nn.MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode, return_indices)  
        self.max_pool_imag = nn.MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode, return_indices) 

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.max_pool_re(x[:,:,:,:,0])
        imaginary = self.max_pool_imag(x[:,:,:,:,1])
        output = torch.stack((real, imaginary), dim=-1)
        
        return output

    
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[:,:,:,:, 0])
        imag = self.bn_im(x[:,:,:,:, 1])
        output = torch.stack([real, imag], dim=-1)
        return output
    

def complex_relu(x):
    
    real = nn.ReLU()(x[:,:,:,:,0])
    imag = nn.ReLU()(x[:,:,:,:,1])
    
    #print(real.shape, imag.shape)
    
    return torch.stack([real, imag], dim=-1)



class ComplexConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, maxpool = True):
        super(ComplexConvBlock, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = stride
        padding = 1
        
        self.conv0 = ComplexConv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = ComplexConv2d(outchannel, outchannel, kernel_size, stride, padding)
        
        self.batch_norm = ComplexBatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = complex_max_pool2d(maxpool_size)
        #self.relu = complex_relu()

        
    def forward(self, x): 
        x = self.conv0(x)
        #print(x.shape)
        x = complex_relu(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = complex_relu(self.maxpool_layer(x))
        else:
            x = complex_relu(x)
        return x, x
    
class ComplexUpConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel,kernel_size=3,stride=1,padding=1, up=True):
        super(ComplexUpConvBlock, self).__init__()
        
        self.up = up
        kernel_size = kernel_size
        stride = stride
        padding = padding
        if self.up == True:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        self.deconv = ComplexConvTranspose2d(inchannel, outchannel, kernel_size, stride, padding)
        self.batch_norm = ComplexBatchNorm2d(outchannel)

        
    def forward(self, x, r):
        if self.up == True:
            x_re = self.upsample(x[:,:,:,:,0])
            x_imag = self.upsample(x[:,:,:,:,1])
            x = torch.stack([x_re, x_imag], dim=-1)
        #print(x.shape, r.shape)
        x = torch.cat([x,r], dim=1)
        #print(x.shape)
        x = self.deconv(x)
        x = self.batch_norm(x)
        x = complex_relu(x)
        
        return x
    
# Stability of a method for multiplying complex matrices with three real matrix multiplications

# We can perform complex matrix multiplication in same way we perform complex multiplication

class ComplexMultihead_Attention(nn.Module):
    def __init__(self, in_ch, embed_dim, num_heads):
        super(ComplexMultihead_Attention, self).__init__()
        
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.split_size = embed_dim // num_heads
        
        self.conv_query =  ComplexConv2d(in_ch, in_ch, 1, 1, padding=0)
        self.conv_key   =  ComplexConv2d(in_ch, in_ch, 1, 1, padding=0)
        self.conv_value =  ComplexConv2d(in_ch, in_ch, 1, 1, padding=0)
        
    def torch_complex_matmul(self, a1, a2):
        
        #print(a1[...,0].shape, a2[...,1].shape)
        
        mul_real = torch.matmul(a1[...,0],a2[...,0]) - torch.matmul(a1[...,1],a2[...,1])
        mul_imag = torch.matmul(a1[...,0],a2[...,1]) + torch.matmul(a1[...,1],a2[...,0])

        return torch.stack([mul_real, mul_imag], dim = -1)

        
        
    def forward(self, x): 
        
        q = self.conv_query(x) # complex
        k = self.conv_key(x) # complex
        v = self.conv_value(x) # complex
        #print(q.shape, k.shape, v.shape)
        
        
        querys = torch.stack(torch.split(q, self.split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(k, self.split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(v, self.split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        #print(querys.shape, keys.shape, values.shape)
        
        #print(querys.shape, keys.transpose(3, 4).shape, 'q.kT')
        
        scores = self.torch_complex_matmul(querys, keys.transpose(3, 4))  # [h, N, T_q, T_k] # complex
        scores = scores / (self.embed_dim ** 0.5)
        #print(scores.shape)
        scores = F.softmax(scores, dim=4) # need to find right dim #TODO
        
        out = self.torch_complex_matmul(scores, values)  # [h, N, T_q, num_units/h] # complex
        
        #print(out.shape)
        
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        return out


class ComplexConvBlock_Self_attention(nn.Module):
    def __init__(self, inchannel, outchannel, embed_dim, num_heads = 8, stride = 1, maxpool = True):
        super(ComplexConvBlock_Self_attention, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = 1
        padding = 1
        
        num_heads = num_heads
        embed_dim = embed_dim
        
        self.conv0 = ComplexConv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = ComplexConv2d(outchannel, outchannel, kernel_size, stride, padding)
        
        self.cma = ComplexMultihead_Attention(outchannel, embed_dim, num_heads)
        
        self.batch_norm = ComplexBatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = complex_max_pool2d(maxpool_size)
        #self.relu = complex_relu()

        
    def forward(self, x): 
        x = self.conv0(x)
        #print(x.shape)
        x = complex_relu(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = complex_relu(self.maxpool_layer(x))
        else:
            x = complex_relu(x)
        #print(x.shape)
        x = self.cma(x)
        return x, x


class ComplexConvBlock_Channel_attention(nn.Module):
    def __init__(self, inchannel, outchannel, embed_dim, num_heads = 8, stride = 1, maxpool = True):
        super(ComplexConvBlock_Channel_attention, self).__init__()
        
        self.maxpool = maxpool
        maxpool_size = (1,2)
        kernel_size = 3
        stride = 1
        padding = 1
        
        num_heads = num_heads
        embed_dim = embed_dim
        
        self.conv0 = ComplexConv2d(inchannel, outchannel, kernel_size, stride, padding)
        self.conv1 = ComplexConv2d(outchannel, outchannel, kernel_size, stride, padding)
        
        self.cma = ComplexMultihead_Attention(outchannel, embed_dim, num_heads)
        
        self.batch_norm = ComplexBatchNorm2d(outchannel)
        if self.maxpool == True:
            self.maxpool_layer = complex_max_pool2d(maxpool_size)
        #self.relu = complex_relu()

        
    def forward(self, x): 
        x = self.conv0(x)
        #print(x.shape)
        x = complex_relu(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        if self.maxpool == True:
            x = complex_relu(self.maxpool_layer(x))
        else:
            x = complex_relu(x)
        #print(x.shape)
        x = x.permute(0,3,2,1,4)
        x = self.cma(x)
        x = x.permute(0,3,2,1,4)
        return x, x


    
class ComplexU_Net(nn.Module):
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(ComplexU_Net, self).__init__()
        
        self.inchannel = inchannel #3
        
        self.conv1 = ComplexConvBlock(inchannel, filters[0])
        self.conv2 = ComplexConvBlock(filters[0], filters[1])
        self.conv3 = ComplexConvBlock(filters[1], filters[2])
        self.conv4 = ComplexConvBlock(filters[2], filters[3])
        self.conv5 = ComplexConvBlock(filters[3], filters[4], maxpool = False)
        
        self.deconv4 = ComplexUpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = ComplexUpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = ComplexUpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = ComplexUpConvBlock(filters[1]+filters[0], filters[0])
        
        self.conv00 = ComplexConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 =    ComplexLinear(256, output_dim)
        self.conv01 = ComplexConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.up_re = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        self.up_imag = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
        self.sigmoid = nn.Tanh()


        
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
        x_re = self.up_re(x[...,0])
        x_imag = self.up_re(x[...,1])
        x = torch.stack([x_re, x_imag], dim=-1)
        
        x = self.conv00(x)
        #print(x.shape)
        x = self.fc0(x)
        #print(x.shape)
        x = self.sigmoid(self.conv01(x))
        #x = self.conv01(x)
        
        return x



class ComplexU_Net_Attention(nn.Module):
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(ComplexU_Net_Attention, self).__init__()
        
        self.inchannel = inchannel #3
        
               
        self.conv1 = ComplexConvBlock_Self_attention(inchannel, filters[0], filters[3])
        self.conv2 = ComplexConvBlock_Self_attention(filters[0], filters[1], filters[2])
        self.conv3 = ComplexConvBlock_Self_attention(filters[1], filters[2], filters[1])
        self.conv4 = ComplexConvBlock_Self_attention(filters[2], filters[3], filters[0])
        self.conv5 = ComplexConvBlock_Self_attention(filters[3], filters[4], filters[0], maxpool = False)
        
        self.deconv4 = ComplexUpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = ComplexUpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = ComplexUpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = ComplexUpConvBlock(filters[1]+filters[0], filters[0])
        
        self.conv00 = ComplexConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 =    ComplexLinear(256, output_dim)
        self.conv01 = ComplexConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.up_re = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        self.up_imag = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
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
        x_re = self.up_re(x[...,0])
        x_imag = self.up_re(x[...,1])
        x = torch.stack([x_re, x_imag], dim=-1)
        
        x = self.conv00(x)
        #print(x.shape)
        x = self.sigmoid(self.fc0(x))
        #print(x.shape)
        x = self.conv01(x)
        
        return x



class ComplexU_Net_Channel_Attention(nn.Module):
    def __init__(self, inchannel=3, filters=[16,32,64, 128, 256], output_dim=257):
        super(ComplexU_Net_Channel_Attention, self).__init__()
        
        self.inchannel = inchannel #3
        
               
        self.conv1 = ComplexConvBlock_Channel_attention(inchannel, filters[0], filters[3])
        self.conv2 = ComplexConvBlock_Channel_attention(filters[0], filters[1], filters[2])
        self.conv3 = ComplexConvBlock_Channel_attention(filters[1], filters[2], filters[1])
        self.conv4 = ComplexConvBlock_Channel_attention(filters[2], filters[3], filters[0])
        self.conv5 = ComplexConvBlock_Channel_attention(filters[3], filters[4], filters[0], maxpool = False)
        
        self.deconv4 = ComplexUpConvBlock(filters[4]+filters[3], filters[3], up=False)
        self.deconv3 = ComplexUpConvBlock(filters[3]+filters[2], filters[2])
        self.deconv2 = ComplexUpConvBlock(filters[2]+filters[1], filters[1])
        self.deconv1 = ComplexUpConvBlock(filters[1]+filters[0], filters[0])
        
        self.conv00 = ComplexConvTranspose2d(filters[0], 1, 1, stride=1, padding=0)
        self.fc0 =    ComplexLinear(256, output_dim)
        self.conv01 = ComplexConvTranspose2d(1, 1, 1, stride=1, padding=0)
        
        self.up_re = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        self.up_imag = nn.UpsamplingBilinear2d(scale_factor=(1,2))
        
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
        x_re = self.up_re(x[...,0])
        x_imag = self.up_re(x[...,1])
        x = torch.stack([x_re, x_imag], dim=-1)
        
        x = self.conv00(x)
        #print(x.shape)
        x = self.sigmoid(self.fc0(x))
        #print(x.shape)
        x = self.conv01(x)
        
        return x







#model = ComplexU_Net_Channel_Attention()
#x = torch.rand(1,3,96,257,2)
#o = model(x)
#o.shape

#model = ComplexU_Net_Attention()
#x = torch.rand(1,3,96,257,2)
#o = model(x)
#o.shape

#model = ComplexU_Net()
#x = torch.rand(1,3,700,257,2)
#o = model(x)
#o.shape


#convblock_catten = ComplexConvBlock_Self_attention(16,64,128)
#x = torch.rand(1,16,128,128,2)
#o,_ = convblock_catten(x)
#o.shape

#convblock_satten = ComplexConvBlock_Self_attention(16,64,128)
#x = torch.rand(1,16,128,128,2)
#o,_ = convblock_satten(x)
#o.shape

#cma = ComplexMultihead_Attention(16, 256, 8)
#x = torch.rand(1,16,64,256,2)
#o = cma(x)
#o.shape

#upconv = ComplexUpConvBlock(192, 16)
#x = torch.rand(1,128,100,32,2)
#r = torch.rand(1,64,100,64,2)
#o = upconv(x,r)
#o.shape

#complex_conv1 = ComplexConvBlock(3, 16)
#x = torch.rand(1,3,100,257,2)
#o,_ = complex_conv1(x)
#o.shape
























































