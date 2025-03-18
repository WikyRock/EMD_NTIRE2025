
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numbers
from einops import rearrange


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)

class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]

class ResBlock(nn.Module):
    def __init__(self, dim, relu_slope=0.2):
        super(ResBlock, self).__init__()
        # Initialize the conv scheme
        self.resconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(relu_slope, inplace=False)
        )

    def forward(self, x):
        out = self.resconv(x)
        out = x + out
        return out

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
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class QkvConv(nn.Module):
    def __init__(self, dim):
        super(QkvConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
    def forward(self, x):
        return self.conv(x)

class ChannelAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(ChannelAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.conv_q = QkvConv(dim)
        self.conv_k = QkvConv(dim)
        self.conv_v = QkvConv(dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        b,c,h,w = x1.shape
        q = self.conv_q(x1)
        k = self.conv_k(x2)
        v = self.conv_v(x2)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.project_out(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, mid_dim):
        super(FeedForwardBlock, self).__init__()
        self.project_in = nn.Conv2d(dim, mid_dim*2, 1, 1, 0)
        self.dwconv = nn.Conv2d(mid_dim*2, mid_dim*2, 3, 1, 1, groups=mid_dim)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(mid_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_dim, mid_dim=None, out_dim=None):
        super(Mlp, self).__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='WithBias'):
        super(CrossAttention, self).__init__()
        self.norm1_x1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_x2 = LayerNorm(dim, LayerNorm_type)
        self.cab = ChannelAttentionBlock(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # nn.LayerNorm(dim)  # 
        self.ffn = FeedForwardBlock(dim, dim*2)  # Mlp(dim, dim*4)  # 

    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        x = x1 + self.cab(self.norm1_x1(x1), self.norm1_x2(x2))
        # x = to_3d(x)
        x = x + self.ffn(self.norm2(x))
        # x = to_4d(x, h, w)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='WithBias'):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.cab = ChannelAttentionBlock(dim, self.num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # nn.LayerNorm(dim)  # 
        self.ffn = FeedForwardBlock(dim, dim*2)  # Mlp(dim, dim*4)  # 

    def forward(self, x):
        b,c,h,w = x.shape
        y = self.norm1(x)
        x = x + self.cab(y, y)
        # x = to_3d(x)
        x = x + self.ffn(self.norm2(x))
        # x = to_4d(x, h, w)
        return x

class DeformableMapping(nn.Module):
    def __init__(self, dim, offset_group=4):
        super(DeformableMapping, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        self.cat_conv = nn.Conv2d(dim*2, dim, 3, 1, 1)
        self.conv_offset_mask = nn.Conv2d(dim, 3*offset_group*kernel_size*kernel_size, kernel_size, stride, padding)
        self.deform_conv2d = torchvision.ops.DeformConv2d(dim, dim, kernel_size, stride, padding, groups=offset_group)

    def forward(self, x1, x2):
        feat = self.cat_conv(torch.cat((x1, x2), 1))
        offset_mask = self.conv_offset_mask(feat)
        offset1, offset2, mask = torch.chunk(offset_mask, 3, 1)
        offset = torch.cat((offset1, offset2), 1)
        mask = torch.sigmoid(mask)
        x = x2 + self.deform_conv2d(x2, offset, mask)
        return x



class Encoder(nn.Module):
    def __init__(self, in_dim_x1=3, in_dim_x2=3, dim=32, heads=[1,2,4], groups=[2,8,16]):
        super(Encoder, self).__init__()
        self.down0_x1 = nn.Sequential(nn.Conv2d(in_dim_x1, dim*(2**0), 3, 1, 1), ResBlock(dim*(2**0)))
        self.down0_x2 = nn.Sequential(nn.Conv2d(in_dim_x2, dim*(2**0), 3, 1, 1), ResBlock(dim*(2**0)))
        self.enc0_dm = DeformableMapping(dim*(2**0), groups[0])
        self.enc0_ca = CrossAttention(dim*(2**0), heads[0])

        self.down1_x1 = nn.Sequential(nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0), ResBlock(dim*(2**1)))
        self.down1_x2 = nn.Sequential(nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0), ResBlock(dim*(2**1)))
        self.enc1_dm = DeformableMapping(dim*(2**1), groups[1])
        self.enc1_ca = CrossAttention(dim*(2**1), heads[1])

        self.down2_x1 = nn.Sequential(nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0), ResBlock(dim*(2**2)))
        self.down2_x2 = nn.Sequential(nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0), ResBlock(dim*(2**2)))
        self.enc2_dm = DeformableMapping(dim*(2**2), groups[2])
        self.enc2_ca = CrossAttention(dim*(2**2), heads[2])

    def forward(self, x1, x2):
        feat_0_x1 = self.down0_x1(x1)
        feat_0_x2 = self.down0_x2(x2)
        feat_0_x2 = self.enc0_dm(feat_0_x1, feat_0_x2)
        feat_0_x1 = self.enc0_ca(feat_0_x1, feat_0_x2)
        
        feat_1_x1 = self.down1_x1(feat_0_x1)
        feat_1_x2 = self.down1_x2(feat_0_x2)
        feat_1_x2 = self.enc1_dm(feat_1_x1, feat_1_x2)
        feat_1_x1 = self.enc1_ca(feat_1_x1, feat_1_x2)

        feat_2_x1 = self.down2_x1(feat_1_x1)
        feat_2_x2 = self.down2_x2(feat_1_x2)
        feat_2_x2 = self.enc2_dm(feat_2_x1, feat_2_x2)
        feat_2_x1 = self.enc2_ca(feat_2_x1, feat_2_x2)

        return feat_0_x1, feat_1_x1, feat_2_x1
    
class Decoder(nn.Module):
    def __init__(self, dim=32, out_dim=3, heads=[1,2,4]):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(dim*(2**2), dim*(2**1), 2, 2)
        self.conv1 = nn.Conv2d(dim*(2**2), dim*(2**1), 1, 1, 0)
        self.dec1 = SelfAttention(dim*(2**1), heads[1])  # ResBlock(dim*(2**1))  # 

        self.up0 = nn.ConvTranspose2d(dim*(2**1), dim*(2**0), 2, 2)
        self.conv0 = nn.Conv2d(dim*(2**1), dim*(2**0), 1, 1, 0)
        self.dec0 = SelfAttention(dim*(2**0), heads[0])  # ResBlock(dim*(2**0))  # 

        self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1)

    def forward(self, feat_0, feat_1, feat_2):
        feats = self.up1(feat_2)
        feats = self.conv1(torch.cat((feats, feat_1), 1))
        feats = self.dec1(feats)

        feats = self.up0(feats)
        feats = self.conv0(torch.cat((feats, feat_0), 1))
        feats = self.dec0(feats)

        pred = self.conv(feats)
        return pred

class DTER(nn.Module):
    def __init__(self, img_dim=3, evt_dim=24, out_dim=3, dim=64):
        super(DTER, self).__init__()
        self.size_adapter = SizeAdapter(minimum_size=32)

        self.enc = Encoder(img_dim, evt_dim, dim)
        self.dec = Decoder(dim, out_dim)

    def forward(self, short, event):
        event = self.size_adapter.pad(event)
       
        short = self.size_adapter.pad(short)

        feat_0, feat_1, feat_2 = self.enc(short, event)
        pred = self.dec(feat_0, feat_1, feat_2) + short

        pred = self.size_adapter.unpad(pred)
        return pred
