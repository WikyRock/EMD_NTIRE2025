
if True:
    import sys
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"#attention the text order

    dir_path = os.path.dirname(os.path.realpath(__file__)) # get now path 
    parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) # get father path

    sys.path.append(parent_dir_path) # add father path

import torch
import math
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

#from basicsr.utils import get_root_logger
import time
from einops import rearrange
import numbers
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .mamba_simple_pan import Mamba

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type,return_to_4d = False):
        super(LayerNorm, self).__init__()
        self.return_to_4d = return_to_4d
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if self.return_to_4d:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(to_3d(x))
    

###############################################################################################
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return y#x * y
    
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

###############################################################################################

   
class CrossMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias',Residual=True):
        super(CrossMamba, self).__init__()
        
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        
        self.LN1 = LayerNorm(dim, LayerNorm_type)
        self.LN2 = LayerNorm(dim, LayerNorm_type)
        self.Residual = Residual
        
    def forward(self,ms,pan): 
        
        #input (B, C, H, W)
        B, C, H, W = ms.shape
        ms = self.LN1(ms)
        pan = self.LN2(pan)
        global_f = self.cross_mamba(ms,inference_params=None, extra_emb=pan) 

        if self.Residual:
            global_f = global_f + ms

         # (B, H*W, C) to (B, C, H, W)
        global_f = global_f.transpose(1, 2).view(B, C, H, W)
        
        return global_f 
    
class DCAM(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias',Residual = True):
        super(DCAM, self).__init__()

        self.DCAM_Blur = CrossMamba(dim = dim, LayerNorm_type = LayerNorm_type, Residual = Residual)
        self.DCAM_Event= CrossMamba(dim = dim, LayerNorm_type = LayerNorm_type, Residual = Residual)

        self.CAB_Blur = CALayer(channel = dim)
        self.CAB_Event = CALayer(channel = dim)
        
        
        
    def forward(self,blur,event): 
        
        blur = self.DCAM_Blur(blur, event)
        event = self.DCAM_Event(event, blur)
        
        ca_blur = self.CAB_Blur(blur)
        ca_event = self.CAB_Event(event)
        
        blur = blur * ca_blur
        event = event * ca_event

        return blur, event

###############################################################################################



def pixel_reshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)

class RdbConv(nn.Module):
    def __init__(self, in_channels, grow_rate, k_size=3):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels, grow_rate, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
    
    
class Rdb(nn.Module):
    def __init__(self, grow_rate0, grow_rate, num_conv_layers, k_size=3):
        super().__init__()

        convs = []
        for c in range(num_conv_layers):
            convs.append(RdbConv(grow_rate0 + c * grow_rate, grow_rate))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.lff = nn.Conv2d(grow_rate0 + num_conv_layers * grow_rate, grow_rate0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.lff(self.convs(x)) + x
    
###############################################################################################

###############################################################################################

class DCA_Mamba_Nores(nn.Module):
    def __init__(self, event_ch=24, sharp_ch=3, depth = 20 , channel = 96,grow_rate= 48):
        super().__init__()

        self.g0 = channel
        self.e_ch = event_ch
        self.b_ch = sharp_ch 
        k_size = 3
        
        self.d = depth
        self.c = 5
        self.g = grow_rate

        self.sharp_ch = sharp_ch
        

        self.e_conv1 = nn.Conv2d(self.e_ch * 4, self.g0, 5, padding=2, stride=1)
        self.e_conv2 = nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)

        self.b_conv1 = nn.Conv2d( self.b_ch * 4, self.g0, 5, padding=2, stride=1)
        self.b_conv2 = nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)

        # Residual dense blocks and dual channel attention
        self.e_rdbs = nn.ModuleList()
        self.b_rdbs = nn.ModuleList()
        self.dcams  = nn.ModuleList()

        
        for i in range(self.d):    

            self.e_rdbs.append(Rdb(grow_rate0=self.g0, grow_rate=self.g, num_conv_layers=self.c))
            self.b_rdbs.append(Rdb(grow_rate0=self.g0, grow_rate=self.g, num_conv_layers=self.c))
            self.dcams.append(DCAM(self.g0))

        # Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(self.d * self.g0, self.g0, 1, padding=0, stride=1),
            nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)
        ])

        self.up_net = nn.Sequential(*[
            nn.Conv2d(self.g0, 256, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, sharp_ch, k_size, padding=(k_size - 1) // 2, stride=1)
        ])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, blur,event):
        
        e_shuffle = pixel_reshuffle(event, 2)
        b_shuffle = pixel_reshuffle(blur, 2)

        e0 = self.e_conv1(e_shuffle)
        e = self.e_conv2(e0)

        b0 = self.b_conv1(b_shuffle)
        b = self.b_conv2(b0)

        rdbs_out = []
 
        for i in range(self.d):
                    
            e = self.e_rdbs[i](e)
            b = self.b_rdbs[i](b)
            b, e = self.dcams[i](b, e)
            
            rdbs_out.append(b)

        x = self.gff(torch.cat(rdbs_out, 1))
        x += b0

        out = self.up_net(x) 
        return out

