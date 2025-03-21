import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"#attention the text order

dir_path = os.path.dirname(os.path.realpath(__file__)) # get now path 
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) # get father path

sys.path.append(parent_dir_path) # add father path

#from NAFNet.NAFNET_arch import NAFBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time

################################################################################################

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
################################################################################################

# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------

class MixChannelAttentionModule(nn.Module):
    def __init__(self, channel=64, reduction=16,feature = 1):
        super(MixChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(feature)
        self.max_pool = nn.AdaptiveMaxPool2d(feature)

     
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y1 = self.conv_fc(y1)
        y2 = self.conv_fc(y2)

        y = self.sigmoid(y1 + y2)

        return y

class PyramidRCABlock(nn.Module):      
    def __init__(self,  out_channels=64, reduction=16,layer = 1):
        super(PyramidRCABlock, self).__init__()


        self.ca = nn.ModuleList()
        self.layer = layer

        for i in range(layer):

            self.ca.append(MixChannelAttentionModule(out_channels, reduction,2**i))


        if self.layer > 1:
      
            self.conv1 = nn.Conv2d(out_channels*layer, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
  

        _, _, h, w = x.size()

        feat_list = []

        for i in range(self.layer):
            feat = F.interpolate(self.ca[i](x), (h, w))
            feat_list.append(feat)

        if self.layer == 1:
            total_feat = feat_list[0]
        else:
            total_feat = self.conv1(torch.cat(feat_list, dim=1))


        return total_feat

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


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, reduction=16, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',  negative_slope=0.2):
        super(RCABlock, self).__init__()
        out_channels = in_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

#        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
#        res = self.res(x)
        #res = self.ca(x)
        #B, C, H, W = res.shape
        #return res.view(B, C, H*W).permute(0, 2, 1) #res + x
        return self.ca(x)

##############################################################

class NAF_Layer(nn.Module):
    def __init__(self, channel = 96):
        super(NAF_Layer, self).__init__()
        
        self.conv = nn.Sequential(*[
            NAFBlock(c=channel),NAFBlock(c=channel)
        ])
        
        self.ca_block = RCABlock(in_channels=channel, out_channels=channel)
    
    def forward(self,x):
        
        res = self.conv(x)
        ca = self.ca_block(x)
        return res*ca
        

##########################################################

def pixel_reshuffle(input, upscale_factor):
    """
    Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.
 
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
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
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

    
class Wiky_FusionWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_e = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_b = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj_e_before_sigmoid = nn.Linear(dim , dim)
        self.proj_b_before_sigmoid = nn.Linear(dim , dim)
        self.sigmoid = nn.Sigmoid()

        self.proj_e = nn.Linear(dim * 2, dim)
        self.proj_drop_e = nn.Dropout(proj_drop)

        self.proj_b = nn.Linear(dim * 2, dim)
        self.proj_drop_b = nn.Dropout(proj_drop)


        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, e, b, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = e.shape

        e_sigmiod = self.sigmoid(self.proj_e_before_sigmoid(e))
        b_sigmiod = self.sigmoid(self.proj_b_before_sigmoid(b))

        qkv_e = self.qkv_e(e).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_e, k_e, v_e = qkv_e[0], qkv_e[1], qkv_e[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_b = self.qkv_b(b).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_b, k_b, v_b = qkv_b[0], qkv_b[1], qkv_b[2]  # make torchscript happy (cannot use tensor as tuple)

        q_e = q_e * self.scale
        q_b = q_b * self.scale

        attn_e = (q_e @ k_e.transpose(-2, -1))
        attn_b = (q_b @ k_b.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn_e = attn_e + relative_position_bias.unsqueeze(0)
        attn_b = attn_b + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn_e = attn_e.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_e = attn_e.view(-1, self.num_heads, N, N)
            attn_e = self.softmax(attn_e)

            attn_b = attn_b.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_b = attn_b.view(-1, self.num_heads, N, N)
            attn_b = self.softmax(attn_b)
        else:
            attn_e = self.softmax(attn_e)
            attn_b = self.softmax(attn_b)

        out_e = (attn_e @ v_e).transpose(1, 2).reshape(B_, N, C)   
        out_b = (attn_b @ v_b).transpose(1, 2).reshape(B_, N, C)


        out_1 = self.proj_e(torch.cat([out_e * e_sigmiod, out_b * e_sigmiod], -1))
        out_1 = self.proj_drop_e(out_1)
 
        out_b = self.proj_b(torch.cat([out_e * b_sigmiod, out_b * b_sigmiod], -1))
        out_b = self.proj_drop_b(out_b)

        return out_1, out_b, None


class FusionSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution=(128, 128), num_heads=6, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.norm1_e = norm_layer(dim)
        self.norm1_b = norm_layer(dim)
        self.attn = Wiky_FusionWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_e = norm_layer(dim)
        self.norm2_b = norm_layer(dim)
        self.mlp_e = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp_b = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, e_f, b_f):
        # e_f: [B, C, H, W]
        # b_f: [B, C, H, W]
        # return:
        B, C, H, W = e_f.shape

        # (B, C, H, W) to (B, H*W, C)
        e_shortcut = e_f.flatten(2).transpose(1, 2)
        b_shortcut = b_f.flatten(2).transpose(1, 2)

        e_f = self.norm1_e(e_shortcut).view(B, H, W, C)
        b_f = self.norm1_b(b_shortcut).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_e_f = torch.roll(e_f, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_b_f = torch.roll(b_f, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_e_f = e_f
            shifted_b_f = b_f

        # partition windows
        e_windows = window_partition(shifted_e_f, self.window_size)  # nW*B, window_size, window_size, C
        e_windows = e_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        b_windows = window_partition(shifted_b_f, self.window_size)  # nW*B, window_size, window_size, C
        b_windows = b_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == (H, W):
            e_attn_windows, b_attn_windows, denoise_feat = self.attn(e_windows, b_windows,
                                                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            e_attn_windows, b_attn_windows, denoise_feat = self.attn(e_windows, b_windows,
                                                                     mask=self.calculate_mask(x_size=(H, W)).to(e_f.device))

        # merge windows
        e_attn_windows = e_attn_windows.view(-1, self.window_size, self.window_size, C)
        b_attn_windows = b_attn_windows.view(-1, self.window_size, self.window_size, C)
        # denoise_feat = denoise_feat.view(-1, self.window_size, self.window_size, C)

        shifted_e_f = window_reverse(e_attn_windows, self.window_size, H, W)  # B H' W' C
        shifted_b_f = window_reverse(b_attn_windows, self.window_size, H, W)  # B H' W' C
        # shifted_denoise_feat = window_reverse(denoise_feat, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            e_f = torch.roll(shifted_e_f, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            b_f = torch.roll(shifted_b_f, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # denoise_feat = torch.roll(shifted_denoise_feat, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            e_f = shifted_e_f
            b_f = shifted_b_f
            # denoise_feat = shifted_denoise_feat

        e_f = e_f.view(B, H * W, C)
        b_f = b_f.view(B, H * W, C)
        # denoise_feat = denoise_feat.view(B, H * W, C)

        # FFN
        e_f = e_shortcut + self.drop_path(e_f)
        b_f = b_shortcut + self.drop_path(b_f)
        e_f = e_f + self.drop_path(self.mlp_e(self.norm2_e(e_f)))
        b_f = b_f + self.drop_path(self.mlp_b(self.norm2_b(b_f)))

        # (B, H*W, C) to (B, C, H, W)
        e_f = e_f.transpose(1, 2).view(B, C, H, W)
        b_f = b_f.transpose(1, 2).view(B, C, H, W)
        # denoise_feat = denoise_feat.transpose(1, 2).view(B, C, H, W)

        return e_f, b_f,  denoise_feat
    
    
###################################################################################################################

###################################################################################################################

###################################################################################################################

class DCAPP_Net(nn.Module):

    def __init__(self, e_ch=24, b_ch=3, out_ch=3, ts_ch=16, base_ch=96, depth=20, c=5, g=48, relu_slope=0.2, window_size=16,continue_re=False, use_cab = True):
        super().__init__()
        self.g0 = base_ch
        # number of RDB blocks, conv layers, out channels
        self.d, self.c, self.g = depth, c, g

        self.use_cab = use_cab
        k_size = 3

        self.e_conv1 = nn.Conv2d(e_ch * 4, self.g0, 5, padding=2, stride=1)
        self.e_conv2 = nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)

        self.b_conv1 = nn.Conv2d(b_ch * 4, self.g0, 5, padding=2, stride=1)
        self.b_conv2 = nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)

        # Residual dense blocks and dense feature fusion
        self.e_rdbs = nn.ModuleList()
        self.b_rdbs = nn.ModuleList()
        self.ca_block_1=  nn.ModuleList()
        self.ca_block_2=  nn.ModuleList()

        self.continue_re = continue_re
        
        for i in range(self.d):    
    
            self.e_rdbs.append(Rdb(grow_rate0=self.g0, grow_rate=self.g, num_conv_layers=self.c))
            self.b_rdbs.append(Rdb(grow_rate0=self.g0, grow_rate=self.g, num_conv_layers=self.c))
            
            if i%2==0 and self.use_cab:
                self.ca_block_1.append(PyramidRCABlock(self.g0, self.g0//6,1))
                self.ca_block_2.append(PyramidRCABlock(self.g0, self.g0//6,1))
                      

        self.window_size = window_size
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.d)]  # stochastic depth decay rule

        self.swin_blocks = nn.ModuleList([
            FusionSwinTransformerBlock(dim=self.g0,
                                       num_heads=6, window_size=self.window_size,
                                       shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                                       mlp_ratio=2.,
                                       qkv_bias=True, qk_scale=None,
                                       drop=0., attn_drop=0.,
                                       drop_path=dpr[i],
                                       norm_layer=nn.LayerNorm)
            for i in range(self.d)])

        self.norm_e = nn.LayerNorm(self.g0)
        self.norm_b = nn.LayerNorm(self.g0)

        # Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(self.d * self.g0, self.g0, 1, padding=0, stride=1),
            nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)
        ])

        self.up_net = nn.Sequential(*[
            nn.Conv2d(self.g0, 256, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3 if not self.continue_re else 32, k_size, padding=(k_size - 1) // 2, stride=1)
        ])
        if self.continue_re:
            self.temporal_decoder = MLP(in_dim=32 + ts_ch, out_dim=out_ch, hidden_list=[256, 256, 256, 256])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, blur,event):
        bs, _, h_b, w_b = blur.shape

        e_shuffle = pixel_reshuffle(event, 2)
        b_shuffle = pixel_reshuffle(blur, 2)

        # pad for partition window
        h_old, w_old = b_shuffle.shape[-2:]
        e_shuffle = self.check_image_size(e_shuffle)
        b_shuffle = self.check_image_size(b_shuffle)

        e0 = self.e_conv1(e_shuffle)
        e = self.e_conv2(e0)

        b0 = self.b_conv1(b_shuffle)
        b = self.b_conv2(b0)

        rdbs_out = []
        
        res_inp_1 = e
        res_inp_2 = b
        
        for i in range(self.d):
                    
            e = self.e_rdbs[i](e)
            b = self.b_rdbs[i](b)
            e, b, _ = self.swin_blocks[i](e, b)
            if i%2 == 1 and self.use_cab:
                ca_1 = self.ca_block_1[i//2](res_inp_1)
                ca_2 = self.ca_block_2[i//2](res_inp_2)
                e = e*ca_1
                b = b*ca_2
                
                res_inp_1 = e
                res_inp_2 = b
            
            rdbs_out.append(b)

        x = self.gff(torch.cat(rdbs_out, 1))
        x += b0
        x = x[:, :, :h_old, :w_old]
        out = self.up_net(x) + blur

        return out




if __name__ == "__main__":
    
    pass