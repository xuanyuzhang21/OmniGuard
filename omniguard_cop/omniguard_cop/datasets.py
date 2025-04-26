import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self, 
                 data_dir, 
                 file_level=2, 
                 regex=None, 
                 secret_len=100, 
                 transform=T.ToTensor()):
        super().__init__()
        if regex is None:
            self.regex = data_dir + '/*' * file_level
        else:
            self.regex = data_dir + regex
        self.images = sorted(glob(self.regex))
        self.secret_len = secret_len
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.transform(img)
        
        sec = torch.zeros(self.secret_len, dtype=torch.float32).random_(2)
        return {
            'image':  img, 
            'secret': sec,
        }