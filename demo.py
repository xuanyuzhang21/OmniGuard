import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model_invert import *
import config as c
# import datasets
import modules.Unet_common as common
import random
from Quantization import Quantization
from PIL import Image
from iml_vit_model import iml_vit_model
import albumentations as albu
from iml_transforms import get_albu_transforms
from albumentations.pytorch import ToTensorV2

ckpt_path = "checkpoint/checkpoint-175.pth"
MaskExtractor = iml_vit_model()

MaskExtractor.load_state_dict(
    torch.load(ckpt_path)['model'],
    strict = True
)
MaskExtractor = MaskExtractor.to(device)
MaskExtractor.eval()


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def equalize_string_lengths(bin_str1, bin_str2):
    # 获取两个字符串的长度
    len1 = len(bin_str1)
    len2 = len(bin_str2)
    
    # 比较长度并根据需要填充'0'
    if len1 < len2:
        # 对bin_str1进行填充
        bin_str1 = bin_str1.zfill(len2)
    elif len2 < len1:
        # 对bin_str2进行填充
        bin_str2 = bin_str2.zfill(len1)
    
    return bin_str1, bin_str2

def calculate_ber(string1, string2):

    string1, string2 = equalize_string_lengths(string1, string2)
    # 检查两个字符串长度是否相等
    if len(string1) != len(string2):
        raise ValueError("两个字符串长度必须相等")
    
    # 检查字符串是否只包含0和1
    if not all(c in '01' for c in string1 + string2):
        raise ValueError("字符串只能包含0和1")
    
    # 计算错位数
    errors = sum(1 for a, b in zip(string1, string2) if a != b)
    
    # 计算总位数
    total_bits = len(string1)
    
    # 计算误码率
    ber = errors / total_bits
    
    return ber
    
def uuidgen(bitlen):

    id = ''.join(random.choice('01') for _ in range(bitlen))
    return id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def load(net, name):
    state_dicts = torch.load(name)
    for k, v in state_dicts['net'].items():
        print(k)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if ('tmp_var' not in k) and ('bm' not in k)}
    net.load_state_dict(network_state_dict, strict=False)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def upsample_if_smaller_than_512(tensor):
    # 获取宽和高 (tensor形状: [batch, channels, height, width])
    _, _, H, W = tensor.shape
    
    while H < 512 or W < 512:
        # 上采样：扩大2倍
        tensor = F.interpolate(tensor, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 更新宽和高
        _, _, H, W = tensor.shape
    
    return tensor

def EditGuard_Hide(net, cover):
    dwt = common.DWT()
    iwt = common.IWT()

    with torch.no_grad():

        b, _, W, H = cover.shape
        image = Image.open("bluesky_white2.png").convert("RGB").resize((H, W))
        result = np.array(image) / 255.
        expanded_matrix = np.expand_dims(result, axis=0) 
        expanded_matrix = np.repeat(expanded_matrix, b, axis=0)
        secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
        secret = secret.permute(0, 3, 1, 2).to(device)

        cover_input = dwt(cover)
        secret_input = dwt(secret)
        # message = torch.randint(2, (1, 64)).to(torch.float32).to(device)
        # print(cover_input.shape, secret_input.shape, message.shape)

        steg_img, output_z, out_temp, secret_temp = net(cover_input, secret_input)

        steg_img = steg_img.permute(0, 2, 3, 1).cpu().squeeze().numpy()
        steg_img = np.clip(steg_img, 0, 1) * 255.0
        steg_img = Image.fromarray(steg_img.astype(np.uint8))

        # torchvision.utils.save_image(steg_img, c.IMAGE_PATH_demo + 'steg.png')

        return steg_img

import torch.nn.functional as F

def downsample_until_smaller_than_1024(tensor):
    _, _, H, W = tensor.shape
    
    while W > 1024 or H > 1024:
        tensor = F.interpolate(tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        _, _, H, W = tensor.shape
    
    return tensor


def EditGuard_Reveal(net, steg):
    # ablu_my = get_albu_transforms(type_='pad', outputsize=(1024, 1024))
    outputsize = 1024
    ablu_my = albu.Compose([
        albu.PadIfNeeded(          
            min_height=outputsize,
            min_width=outputsize, 
            border_mode=0, 
            value=0, 
            position='top_left',
            mask_value=0),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Crop(0, 0, outputsize, outputsize),
        ToTensorV2()
    ])

    dwt = common.DWT()
    iwt = common.IWT()

    with torch.no_grad():

        # b, _, W, H = steg.shape

        steg_input = dwt(steg)
        image_fuse = steg

        output_image = net(steg_input.to(device), rev=True)
        secret_rev = output_image.narrow(1, 0, 12)
        secret_rev = iwt(secret_rev)

        secret_rev = secret_rev.to(device)

        artifact, fuse = secret_rev.to(device), image_fuse.to(device)
        
        artifact = downsample_until_smaller_than_1024(artifact)
        fuse = downsample_until_smaller_than_1024(fuse)

        print(artifact.shape, fuse.shape)
        b, _, W, H = artifact.shape

        artifact = artifact.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255
        fuse = fuse.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255

        artifact = ablu_my(image=artifact)['image'].to(device).unsqueeze(0)
        fuse = ablu_my(image=fuse)['image'].to(device).unsqueeze(0)

        mask_pred = MaskExtractor(artifact, fuse)
        mask_pred = mask_pred[:, :, 0:W, 0:H]

        torchvision.utils.save_image(mask_pred, c.IMAGE_PATH_demo + 'residual.png')

        return mask_pred
    


if __name__ == "__main__":
    str1, message = EditGuard_Hide()
    _, recmessage = EditGuard_Reveal(str1)
    
    recmessage = (recmessage > 0).int()

    message = (message > 0).int()

    # 将结果转换为 Python 列表，然后连接成一个字符串
    binary_string = ''.join(str(x.item()) for x in recmessage.flatten())
    print(binary_string)
    message = ''.join(str(x.item()) for x in message.flatten())
    print(message)
    print(1 - calculate_ber(message, binary_string))


