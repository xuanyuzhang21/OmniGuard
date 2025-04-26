import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.color as color
from focal_frequency_loss import FocalFrequencyLoss

from .lpips import *


def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class ImageSecretLoss(nn.Module):
    def __init__(self, 
                 pixel_loss='l2',
                 color_space='yuv',
                 color_w=None,
                 lpips_net='alex', 
                 lpips_use_dropout=True):
        super().__init__()
        name_to_loss = {
            "l1": l1,
            "l2": l2,
        }
        assert pixel_loss in name_to_loss.keys()
        assert color_space in ["yuv", "rgb"]
        self.pixel_loss = name_to_loss[pixel_loss]
        self.color_space = color_space
        if color_w is None:
            if color_space == 'yuv': color_w = np.array([1, 100, 100])
            else: color_w = np.array([0.6378, 2.1456, 0.2166]) * 64
        self.color_w = torch.FloatTensor(color_w).view(1, 3, 1, 1).contiguous()
        
        self.frequency_loss = FocalFrequencyLoss().eval()
        self.perceptual_loss = LPIPS(lpips_net, lpips_use_dropout).eval()
        
        self.secret_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                inputs, reconstructions, image_w=None,
                secrets=None, secrets_logit=None, secret_w=None,
                pixel_w=1.5, frequency_w=1.5, perceptual_w=1.0):
        y_rec, y_ori = reconstructions * 0.5 + 0.5, inputs * 0.5 + 0.5
        if self.color_space == 'yuv':
            y_rec, y_ori = color.rgb_to_yuv(y_rec), color.rgb_to_yuv(y_ori)
        pixel_loss = self.pixel_loss(y_rec.contiguous(), y_ori.contiguous())
        color_w = self.color_w.to(pixel_loss.device)
        pixel_loss = (color_w * pixel_loss).sum(dim=1).mean()
        
        rec_loss = pixel_w * pixel_loss
        if frequency_w > 0.0:
            frequency_loss = self.frequency_loss(
                reconstructions.contiguous(), inputs.contiguous())
            rec_loss = rec_loss + frequency_w * frequency_loss
        else:
            frequency_loss = torch.zeros(size=())
        if perceptual_w > 0.0:
            perceptual_loss = self.perceptual_loss(
                reconstructions.contiguous(), inputs.contiguous())  # input, target
            rec_loss = rec_loss + perceptual_w * perceptual_loss
        else:
            perceptual_loss = torch.zeros(size=())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        image_loss = nll_loss
        secret_loss = self.secret_loss(secrets_logit, secrets)
        
        loss = image_w * image_loss + secret_w * secret_loss
        # TODO check whether norm is necessary
        loss = loss / (image_w + secret_w)
        
        loss_dict = {
            # loss
            'loss':             loss.item(),
            'image_loss':       image_loss.item(),
            'secret_loss':      secret_loss.item(),
            'pixel_loss':       pixel_loss.mean().item(),
            'frequency_loss':   frequency_loss.item(),
            'perceptual_loss':  perceptual_loss.mean().item(),
            # weight
            'image_w':          image_w,
            'secret_w':         secret_w,
            'pixel_w':          pixel_w,
            'frequency_w':      frequency_w,
            'perceptual_w':     perceptual_w,
        }
        return loss, loss_dict
