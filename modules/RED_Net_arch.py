import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"#attention the text order

dir_path = os.path.dirname(os.path.realpath(__file__)) # get now path 
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) # get father path

sys.path.append(parent_dir_path) # add father path


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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


class RED_Net(nn.Module):
    def __init__(self, event_ch=24, sharp_ch=3):
        super().__init__()
        self.g0 = 96
        k_size = 3

        # number of RDB blocks, conv layers, out channels
        self.d = 20
        self.c = 5
        self.g = 48

        # Shallow feature extraction net
        self.sfe_1 = nn.Conv2d((sharp_ch + event_ch) * 4, self.g0, 5, padding=2, stride=1)
        self.sfe_2 = nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)

        # Residual dense blocks and dense feature fusion
        self.rdbs = nn.ModuleList()
        for i in range(self.d):
            self.rdbs.append(
                Rdb(grow_rate0=self.g0, grow_rate=self.g, num_conv_layers=self.c)
            )

        # Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(self.d * self.g0, self.g0, 1, padding=0, stride=1),
            nn.Conv2d(self.g0, self.g0, k_size, padding=(k_size - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.up_net = nn.Sequential(*[
            nn.Conv2d(self.g0, 256, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, sharp_ch, k_size, padding=(k_size - 1) // 2, stride=1)
        ])

    def forward(self, sharp_img, event_img):
        """
        args
            event_img: [bs, c, h, w]
            sharp_img :[bs, c, h, w]
            ts: list of [bs, c]
        """
        B, C, H, W = event_img.shape

        b_shuffle = pixel_reshuffle(torch.cat((sharp_img, event_img), 1), 2)

        f_1 = self.sfe_1(b_shuffle)
        x = self.sfe_2(f_1)

        rdbs_out = []
        for i in range(self.d):
            x = self.rdbs[i](x)
            rdbs_out.append(x)

        x = self.gff(torch.cat(rdbs_out, 1))
        x += f_1

        out = self.up_net(x) + sharp_img
        

        
        
        return out
    

if __name__ == "__main__":

    if (1):
        print("cuda.is_available", torch.cuda.is_available())
        print("GPU device_count", torch.cuda.device_count())
        print("torch version.cuda", torch.version.cuda)
        print("GPU current_device", torch.cuda.current_device())
        print("GPU get_device_name", torch.cuda.get_device_name())
        
    if True: 
        net = RED_Net()
        net = net.cuda()

        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('the number of network parameters: {}'.format(total_params))
    if True:
        batch = 1

        H ,W = 180,320

        ev3 = torch.rand((batch, 24, H, W)).float().cuda()
        img = torch.rand((batch, 3, H, W)).float().cuda()

        now_time = time.time()

        # out = net(img,ev3)
        # print(time.time() - now_time)
        # print(out[0].shape)
        
        
        from thop import profile
   
    
        # checkpoint = torch.load('/home_origin/ChengZY/LuoWeiqi/Demo_IVF_Lite/Train_SSM/models/DCAM_Ver1/models/checkpoint-epoch300.pth')
        # #net.load_state_dict(checkpoint['state_dict'])
        # net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        
        
        w,h = 1632 , 1224
    

        img = torch.ones(1, 3, h,w).cuda()
        events = torch.ones(1, 24, h,w).cuda()
        flops, params = profile(net, (img,events))
        print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
        total = sum(p.numel() for p in net.parameters())
        print("Total params: %.2fM" % (total/1e6))   
