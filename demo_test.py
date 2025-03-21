import os
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from test_dataloader import Dataloader_test as Dataloader_val
from modules import define_network
from losses import Criterion,l1Loss
from utilities.checkpoint import Saver
from utilities.data_process import ensure_dir, event_plot_cuda


from utilities.evaluation import calculate_psnr,calculate_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def ensure_dir(s):
    if not os.path.exists(s):
        os.makedirs(s)

def trainer(args, data_loader_train):
    """ -------------------- build criterion -------------------- """

    """ -------------------- build Net -------------------- """
    net = define_network(args)()
    net = nn.DataParallel(net).cuda()
    net = net.eval()
    for _,param in net.named_parameters():
        param.requires_grad = False


    net.load_state_dict(torch.load(args.net_path), strict=False)
    #net.load_state_dict(torch.load(args.net_path)['state_dict'])
    print("Load pretrained net from " + args.net_path)

    if args.save_image:
        ensure_dir(args.save_image_path)

    Record = []
    iter = 0
    PSNR, SSIM , LPIPS, NEED_TIME = 0., 0., 0.,0.
    """ -------------------- start test -------------------- """
    print("Start test...")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]  


    tbar = tqdm(data_loader_train)
    for idx, (blur, event, prefix) in enumerate(tbar):
        """ -------------------- load to GPU -------------------- """
  
        blur = blur.cuda()
       
        event = event.cuda()

        small_time= time.time()
     

        preds = net(blur, event)

  
        NEED_TIME += time.time()-small_time



        if  isinstance(preds, list):
            preds = preds[1].detach().cpu().numpy().squeeze(axis=0).clip(0,1)
           
        else:
            preds = preds.detach().cpu().numpy().squeeze(axis=0).clip(0,1)

        preds = (preds*255).transpose(1,2,0)


        if args.save_image:
            
            pred_shape_image_npy_1 = preds.astype(np.uint8) 

            if args.if_compress:  

                success = cv2.imencode('.jpg', pred_shape_image_npy_1, encode_param)[1]  
                if success is not None:  
                    with open(args.save_image_path + "/" + prefix[0] + ".png", 'wb') as f:  
                        f.write(success)  
            
            else:     
                cv2.imwrite(args.save_image_path + "/" + prefix[0] + ".png",pred_shape_image_npy_1) 

        iter += 1


        
    return  NEED_TIME/iter

def main(args):

    """ -------------------- load dataset -------------------- """
    dataset_train = Dataloader_val(args)
    data_loader_train = DataLoader(dataset_train, batch_size=1,shuffle=False)
    
    """ -------------------- build Net -------------------- """
    d = trainer(args, data_loader_train)

    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")

    #you need to change the path of dataset according to your own dataset

    parser.add_argument("--dataset_path", type=str, default="/home_origin/ChengZY/LuoWeiqi/NTIRE2025/", help="data path")
    
    
    
    parser.add_argument("--dataset_name", type=str, default="HighREV", help="data path")
    parser.add_argument("--save_image", type=bool, default=True, help="whether to save image")

    parser.add_argument("--save_image_path", type=str, default="./test_res")
    parser.add_argument("--arch", type=str, default="DCA_Mamba_Nores") 
    parser.add_argument("--net_path", type=str, default="models/6_DCA_Mamba_Nores.pth", help="pretrained model path")
    parser.add_argument("--event_type", type=str, default="pure_events_to_voxel_grid")
        
    parser.add_argument("--test_range", type=int, default=1)

    parser.add_argument("--if_compress", type=bool, default="False")

    args = parser.parse_args()
    d = main(args)

    print("TIME: %.6fs" % (d))

   