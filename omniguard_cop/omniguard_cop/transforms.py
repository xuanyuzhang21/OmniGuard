from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import AutoencoderKL


class VAE(nn.Module):
    def __init__(self, 
                 model_name='CompVis/stable-diffusion-v1-4'):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.vae.eval()

    def forward(self, x):
        """
        :param x: input image in [-1, 1]
        :return: output image in [-1, 1]
        """
        z = self.vae.encode(x).latent_dist.sample()
        x = self.vae.decode(z).sample
        return x
    
    
class RandomWatermarkRemoval(nn.Module):
    def __init__(self, 
                 data_dir, 
                 file_level=1):
        super().__init__()
        self.regex = data_dir + '/*' * file_level
        self.masks = sorted(glob(self.regex))
        self.transform = T.ToTensor()

    def random_mask_idx(self):
        mask_idx = torch.zeros((), dtype=torch.long).random_(0, len(self.masks)).item()
        return mask_idx
        
    def forward(self, x, x_uwm, mask_idx=None):
        h, w = x.size()[2:]  # b, c, h, w
        if mask_idx is None: mask_idx = self.random_mask_idx()
        mask = Image.open(self.masks[mask_idx]).convert('L')
        mask = mask.resize((w, h), resample=Image.NEAREST)
        
        mask = self.transform(mask).unsqueeze(dim=0)  # 1, 1, h, w
        mask = mask.to(x.device)
        return x_uwm * mask + x * (1 - mask)


class ImagePerturbator(nn.Module):
    def __init__(self, 
                 base_perturb, 
                 opt_perturbs,
                 opt_nprobs=[0.1, 0, 0.9],
                 # TODO check whether following config is better
                 # opt_nprobs=[0.1, 0.3, 0.6],
                 opt_pprobs=None):
        super().__init__()
        self.base_perturb = base_perturb
        self.opt_perturbs = opt_perturbs
        self.opt_nprobs = torch.FloatTensor(opt_nprobs)
        opt_pprobs = [1] * len(opt_perturbs) if opt_pprobs is None else opt_pprobs
        self.opt_pprobs = torch.FloatTensor(opt_pprobs)
        self.prev_idxs = None
        self.cur_idxs = None
        
    def random_opt_num(self):
        opt_num = torch.multinomial(self.opt_nprobs, num_samples=1).item()
        return opt_num
        
    def random_perturb_idxs(self, opt_num=None):
        if opt_num is None: opt_num = self.random_opt_num()
        if opt_num > 0: perturb_idxs = torch.multinomial(
            self.opt_pprobs, num_samples=opt_num)
        else: perturb_idxs = torch.LongTensor()
        return perturb_idxs
        
    def forward(self, x, perturb_w=1.0, perturb_idxs=None):
        """
        :param x: input image in [-1, 1]
        :param perturb_w: perturbation weight in output image 
        :return: output image in [-1, 1]
        """
        assert perturb_w >= 0 and perturb_w <= 1, f"Invalid perturb_w: {perturb_w}"
        x = x * 0.5 + 0.5 # [-1, 1] -> [0, 1]

        # base perturbations
        if self.base_perturb is not None: x = self.base_perturb(x)

        # optional perturbations
        x_perturbed = x * 1.0
        if perturb_idxs is None: perturb_idxs = self.random_perturb_idxs()
        for perturb_idx in perturb_idxs:
            x_complete = self.opt_perturbs[perturb_idx.item()](x_perturbed)
            x_perturbed = perturb_w * x_complete + (1 - perturb_w) * x_perturbed
        
        x = x * 2 - 1 # [0, 1] -> [-1, 1]
        return x