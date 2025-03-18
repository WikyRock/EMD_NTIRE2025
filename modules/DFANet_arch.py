import torch
import torch.nn as nn

from modules.arch_util import SizeAdapter, ResBlock, CrossModalAttention, SameModalAttention


class Encoder(nn.Module):
    def __init__(self, in_dim_x1=3, in_dim_x2=3, dim=32, heads=[1,2,4,8], groups=[1,2,2,3]):
        super(Encoder, self).__init__()
        self.groups = groups

        self.down0_x1 = nn.Conv2d(in_dim_x1, dim*(2**0), 3, 1, 1)
        self.down0_x2 = nn.Conv2d(in_dim_x2, dim*(2**0), 3, 1, 1)
        self.enc0 = nn.ModuleList()
        for i in range(groups[0]):
            self.enc0.append(CrossModalAttention(dim*(2**0), heads[0]))

        self.down1_x1 = nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0)
        self.down1_x2 = nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0)
        self.enc1 = nn.ModuleList()
        for i in range(groups[1]):
            self.enc1.append(CrossModalAttention(dim*(2**1), heads[1]))

        self.down2_x1 = nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0)
        self.down2_x2 = nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0)
        self.enc2 = nn.ModuleList()
        for i in range(groups[2]):
            self.enc2.append(CrossModalAttention(dim*(2**2), heads[2]))

        self.down3_x1 = nn.Conv2d(dim*(2**2), dim*(2**3), 2, 2, 0)
        self.down3_x2 = nn.Conv2d(dim*(2**2), dim*(2**3), 2, 2, 0)
        self.enc3 = nn.ModuleList()
        for i in range(groups[3]):
            self.enc3.append(CrossModalAttention(dim*(2**3), heads[3]))

    def forward(self, x1, x2):
        feat_0_x1 = self.down0_x1(x1)
        feat_0_x2 = self.down0_x2(x2)
        for i in range(self.groups[0]):
            feat_0_x1, feat_0_x2 = self.enc0[i](feat_0_x1, feat_0_x2)
        
        feat_1_x1 = self.down1_x1(feat_0_x1)
        feat_1_x2 = self.down1_x2(feat_0_x2)
        for i in range(self.groups[1]):
            feat_1_x1, feat_1_x2 = self.enc1[i](feat_1_x1, feat_1_x2)

        feat_2_x1 = self.down2_x1(feat_1_x1)
        feat_2_x2 = self.down2_x2(feat_1_x2)
        for i in range(self.groups[2]):
            feat_2_x1, feat_2_x2 = self.enc2[i](feat_2_x1, feat_2_x2)

        feat_3_x1 = self.down3_x1(feat_2_x1)
        feat_3_x2 = self.down3_x2(feat_2_x2)
        for i in range(self.groups[3]):
            feat_3_x1, feat_3_x2 = self.enc3[i](feat_3_x1, feat_3_x2)

        return feat_0_x1, feat_1_x1, feat_2_x1, feat_3_x1
    
class Decoder(nn.Module):
    def __init__(self, dim=32, out_dim=3, heads=[1,2,4,8], groups=[1,2,2,3]):
        super(Decoder, self).__init__()
        self.up2 = nn.ConvTranspose2d(dim*(2**3), dim*(2**2), 2, 2)
        self.conv2 = nn.Conv2d(dim*(2**3), dim*(2**2), 1, 1, 0)
        self.dec2 = nn.Sequential(*[SameModalAttention(dim*(2**2), heads[2]) for i in range(groups[2])])

        self.up1 = nn.ConvTranspose2d(dim*(2**2), dim*(2**1), 2, 2)
        self.conv1 = nn.Conv2d(dim*(2**2), dim*(2**1), 1, 1, 0)
        self.dec1 = nn.Sequential(*[SameModalAttention(dim*(2**1), heads[1]) for i in range(groups[1])])

        self.up0 = nn.ConvTranspose2d(dim*(2**1), dim*(2**0), 2, 2)
        self.conv0 = nn.Conv2d(dim*(2**1), dim*(2**0), 1, 1, 0)
        self.dec0 = nn.Sequential(*[SameModalAttention(dim*(2**0), heads[0]) for i in range(groups[0])])

        self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1)

    def forward(self, feat_0, feat_1, feat_2, feat_3):
        feats = self.up2(feat_3)
        feats = self.conv2(torch.cat((feats, feat_2), 1))
        feats = self.dec2(feats)

        feats = self.up1(feat_2)
        feats = self.conv1(torch.cat((feats, feat_1), 1))
        feats = self.dec1(feats)

        feats = self.up0(feats)
        feats = self.conv0(torch.cat((feats, feat_0), 1))
        feats = self.dec0(feats)

        pred = self.conv(feats)
        return pred
    
class DFANet(nn.Module):
    def __init__(self, img_dim=3, evt_dim=6, out_dim=3, dim=64):
        super(DFANet, self).__init__()
        self.size_adapter = SizeAdapter(minimum_size=32)

        self.enc = Encoder(img_dim, evt_dim, dim)
        self.dec = Decoder(dim, out_dim)

    def forward(self, long, event):
        event = self.size_adapter.pad(event)
        long = self.size_adapter.pad(long)

        feat_0, feat_1, feat_2, feat_3 = self.enc(long, event)
        pred = self.dec(feat_0, feat_1, feat_2, feat_3) + long

        pred = self.size_adapter.unpad(pred)
        return [pred]