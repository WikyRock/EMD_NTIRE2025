import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Criterion(nn.Module):
    """
    Compute loss and evaluation metrics
    """
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.args = args
        self.weights = args.loss_weight

    def calc_psnr_loss(self, preds, true):
        scale = 10 / np.log(10)
        loss = 0.
        for pred in preds:
            loss += scale * torch.log(((pred - true) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        return loss
    
    def forward(self, preds, true):
        loss_psnr = self.calc_psnr_loss(preds, true)

        """ -------------------- aggregate losses -------------------- """
        l_sum = self.weights[0]*loss_psnr
        loss_list = [self.weights[0]*loss_psnr]

        return l_sum, loss_list
    

def l1Loss(output, target):
    l1_loss = nn.L1Loss()
    l1 = l1_loss(output, target)
    return l1
