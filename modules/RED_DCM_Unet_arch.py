if True:
    import sys
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"#attention the text order

    dir_path = os.path.dirname(os.path.realpath(__file__)) # get now path 
    parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) # get father path

    sys.path.append(parent_dir_path) # add father path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from modules.mamba_simple_pan import Mamba
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .mamba_simple_pan import Mamba

##########################################################################


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

#########################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size,  relu_slope=0.1): # cat
        super(ResidualBlock, self).__init__()
     
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

    def forward(self, x):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)


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

class DualCrossMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias',Residual = True):
        super(DualCrossMamba, self).__init__()

        self.DCAM_Blur = CrossMamba(dim = dim, LayerNorm_type = LayerNorm_type, Residual = Residual)
        self.DCAM_Event= CrossMamba(dim = dim, LayerNorm_type = LayerNorm_type, Residual = Residual)
        
    def forward(self,blur,event): 
        
        blur_mid = self.DCAM_Blur(blur, event)
        event_mid = self.DCAM_Event(event, blur)


        return blur_mid, event_mid
    
class DualCrossMamba_list(nn.Module):
    def __init__(self, dim,layer =1, LayerNorm_type='WithBias',Residual = True,Need_RDB = 0):
        super(DualCrossMamba_list, self).__init__()

        self.layer = layer
        self.dcams  = nn.ModuleList()
        for i in range(layer):    
            self.dcams.append(DualCrossMamba(dim,LayerNorm_type,Residual))

        self.need_rdb = Need_RDB

        if self.need_rdb:
            self.rdbs_e = nn.ModuleList()
            self.rdbs_b = nn.ModuleList()
            for i in range(self.layer):
                self.rdbs_e.append(
                    Rdb(grow_rate0=dim, grow_rate=dim//2, num_conv_layers=self.need_rdb)
                )
                self.rdbs_b.append(
                    Rdb(grow_rate0=dim, grow_rate=dim//2, num_conv_layers=self.need_rdb)
                )


    def forward(self,blur,event): 

        for i in range(self.layer):

            if self.need_rdb:
                blur = self.rdbs_b[i](blur)
                event = self.rdbs_e[i](event)
            blur,event = self.dcams[i](blur,event)

        return blur, event




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################

class RED_DCM_Unet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        eve_channels=24,
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(RED_DCM_Unet, self).__init__()

        self.patch_embed_b = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_e = OverlapPatchEmbed(eve_channels, dim)

     
        self.encoder_level1  = DualCrossMamba_list(dim = dim*2**0, layer = num_blocks[0],Need_RDB = 5)
       
        
        self.down1_2_b = Downsample(dim) ## From Level 1 to Level 2
        self.down1_2_e = Downsample(dim) 
        self.encoder_level2  = DualCrossMamba_list(dim = dim*2**1, layer = num_blocks[1],Need_RDB = 5)
        
        self.down2_3_b = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.down2_3_e = Downsample(int(dim*2**1))
        self.encoder_level3 = DualCrossMamba_list(dim = dim*2**2, layer = num_blocks[2])
       

        self.down3_4_b = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.down3_4_e = Downsample(int(dim*2**2))
        self.encoder_level4 = DualCrossMamba_list(dim = dim*2**3, layer = num_blocks[3])
        

        self.up4_3_b = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.up4_3_e = Upsample(int(dim*2**3))
        self.reduce_chan_level3_b = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level3_e = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = DualCrossMamba_list(dim = dim*2**2, layer = num_blocks[2])
       

        self.up3_2_e = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.up3_2_b = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2_e = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_b = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = DualCrossMamba_list(dim = dim*2**1, layer = num_blocks[1],Need_RDB = 5)
        

        self.up2_1_e = Upsample(int(dim*2**1)) ## From Level 2 to Level 1
        self.up2_1_b = Upsample(int(dim*2**1)) # (NO 1x1 conv to reduce channels)
        self.decoder_level1 = DualCrossMamba_list(dim = dim*2**1, layer = num_blocks[0],Need_RDB = 5)
      

        self.refinement =CrossMamba(dim = dim*2**1 )
        #### For Dual-Pixel Defocus Deblurring Task ####

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        if h % window_size != 0 or w % window_size != 0:
            mod_pad_h = (window_size - h % window_size) % window_size
            mod_pad_w = (window_size - w % window_size) % window_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img,evp_img):


        h_old, w_old = inp_img.shape[-2:]
        inp_img = self.check_image_size(inp_img)
        evp_img = self.check_image_size(evp_img)



        inp_enc_level1_b = self.patch_embed_b(inp_img)
        inp_enc_level1_e = self.patch_embed_e(evp_img)
        out_enc_level1_b , out_enc_level1_e  = self.encoder_level1(inp_enc_level1_b,inp_enc_level1_e)
        
        

        inp_enc_level2_b = self.down1_2_b(out_enc_level1_b)
        inp_enc_level2_e = self.down1_2_e(out_enc_level1_e)
        out_enc_level2_b , out_enc_level2_e = self.encoder_level2(inp_enc_level2_b,inp_enc_level2_e)
         

        inp_enc_level3_b = self.down2_3_b(out_enc_level2_b)
        inp_enc_level3_e = self.down2_3_e(out_enc_level2_e)
        out_enc_level3_b , out_enc_level3_e = self.encoder_level3(inp_enc_level3_b,inp_enc_level3_e)
  

        inp_enc_level4_b = self.down3_4_b(out_enc_level3_b)
        inp_enc_level4_e = self.down3_4_e(out_enc_level3_e)
        latent_b , latent_e  = self.encoder_level4(inp_enc_level4_b,inp_enc_level4_e)
        

        
        inp_dec_level3_b = self.up4_3_b(latent_b)
        inp_dec_level3_e = self.up4_3_e(latent_e)
        inp_dec_level3_b = torch.cat([inp_dec_level3_b, out_enc_level3_b], 1)
        inp_dec_level3_e = torch.cat([inp_dec_level3_e, out_enc_level3_e], 1)
        inp_dec_level3_b = self.reduce_chan_level3_b(inp_dec_level3_b)
        inp_dec_level3_e = self.reduce_chan_level3_e(inp_dec_level3_e)
        out_dec_level3_b, out_dec_level3_e = self.decoder_level3(inp_dec_level3_b,inp_dec_level3_e)

        inp_dec_level2_b = self.up3_2_b(out_dec_level3_b)
        inp_dec_level2_e = self.up3_2_e(out_dec_level3_e)
        inp_dec_level2_b = torch.cat([inp_dec_level2_b, out_enc_level2_b], 1)
        inp_dec_level2_e = torch.cat([inp_dec_level2_e, out_enc_level2_e], 1)
        inp_dec_level2_b = self.reduce_chan_level2_b(inp_dec_level2_b)
        inp_dec_level2_e = self.reduce_chan_level2_e(inp_dec_level2_e)
        out_dec_level2_b, out_dec_level2_e = self.decoder_level2(inp_dec_level2_b,inp_dec_level2_e)

        inp_dec_level1_b = self.up2_1_b(out_dec_level2_b)
        inp_dec_level1_e = self.up2_1_e(out_dec_level2_e)
        inp_dec_level1_b = torch.cat([inp_dec_level1_b, out_enc_level1_b], 1)
        inp_dec_level1_e = torch.cat([inp_dec_level1_e, out_enc_level1_e], 1)
        out_dec_level1_b,out_dec_level1_e= self.decoder_level1(inp_dec_level1_b,inp_dec_level1_e)
   
        out_dec_level1 = self.refinement(out_dec_level1_b,out_dec_level1_e)

        out_dec_level1 = self.output(out_dec_level1) + inp_img


        out_dec_level1 = out_dec_level1[:, :, :h_old, :w_old]


        return out_dec_level1
    

if __name__ == '__main__':
    
    if (1):
        print("cuda.is_available", torch.cuda.is_available())
        print("GPU device_count", torch.cuda.device_count())
        print("torch version.cuda", torch.version.cuda)
        print("GPU current_device", torch.cuda.current_device())
        print("GPU get_device_name", torch.cuda.get_device_name())
        
    if True: 
        net = RED_DCM_Unet()
        net = net.cuda()

        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('the number of network parameters: {}'.format(total_params))
    if True:
        batch = 1

        H ,W = 180,320

        ev3 = torch.ones((batch, 24, H, W)).float().cuda()
        img = torch.ones((batch, 3, H, W)).float().cuda()

       

    from thop import profile
   

    flops, params = profile(net, (img,ev3))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2fM" % (total/1e6))  
