import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
import random
from Quantization import Quantization
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from JPEG import DiffJPEG
from jpegtest import JpegTest

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()


TM_SCHEMA_CODE=TrustMark.Encoding.BCH_5
tm=TrustMark(verbose=True, model_type='Q', encoding_type=TM_SCHEMA_CODE)
bitlen=tm.schemaCapacity()
print(bitlen)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "/data/zhangxuanyu/sd_inpaint",
                torch_dtype=torch.float16,
            ).to("cuda")

with torch.no_grad():
    psnr_s = []
    psnr_c = []
    bit_acc = []
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        # cover = data[data.shape[0] // 2:, :, :, :]
        cover = data

        b, _, H, W = cover.shape

        # secret = cover
        R = 0
        G = 0
        B = 0
        # image = Image.new('RGB', (W, H), (R, G, B))
        image = Image.open("/data/zhangxuanyu/HiNet/bluesky_white2.png").convert("RGB").resize((W, H))
        # image = Image.open("/data/zhangxuanyu/HiNet/bluesky.jpg").convert("RGB").resize((W, H))
        # image = Image.open("/data/zhangxuanyu/HiNet/blueskyyy.png").convert("RGB").resize((W, H))
        result = np.array(image) / 255.
        expanded_matrix = np.expand_dims(result, axis=0) 
        expanded_matrix = np.repeat(expanded_matrix, b, axis=0)
        secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
        secret = secret.permute(0, 3, 1, 2).to(device)

        cover_input = dwt(cover)
        # secret_input = dwt(secret)
        secret_input = dwt(cover)
        input_img = torch.cat((cover_input, secret_input), 1)

        bitlen=tm.schemaCapacity()
        # message = uuidgen(68)
        message = uuidgen(bitlen)

        steg_img, output_z, out_temp = net(input_img, message)

        quant = Quantization()

        import random
        # choice = random.randint(0, 1)
        choice = 2
            
        if choice == 0:
            NL = float((np.random.randint(1, 10))/255)
            noise = np.random.normal(0, NL, steg_img.shape)
            torchnoise = torch.from_numpy(noise).cuda().float()

            steg_img_tmp = steg_img.clone() + torchnoise
            steg_decode = quant(steg_img_tmp)
        
        elif choice == 1:
            NL = int(np.random.randint(70,95))
            DiffJPEG_opt = JpegTest(NL, 0)
            steg_decode = DiffJPEG_opt(steg_img)
        
        elif choice == 2:
            steg_decode = quant(steg_img)            

        # print(choice, NL)

        import random
        from PIL import Image
        import os

        b, _, h, w = steg_decode.shape

        # masksrc = "/data/zhangxuanyu/segmentanything/maskcocobig/"
        masksrc = "/data/zhangxuanyu/segmentanything/cocomaskfinal/"
        mask_image = Image.open(masksrc + str(i+1).zfill(4) + ".png").convert("L")
        mask_image = mask_image.resize((w, h))
        h, w = mask_image.size
        
        image = steg_decode.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
        prompt = ""
        image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
        image_inpaint = pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
        image_inpaint = np.array(image_inpaint) / 255.
        mask_image = np.array(mask_image)
        mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
        mask_image = mask_image.astype(np.uint8)
        image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
        image_fuse = torch.from_numpy(image_fuse).permute(2,0,1).unsqueeze(0).float().cuda()


        # mask_image = Image.open(f"/data/zhangxuanyu/segmentanything/maskcocotest/{i+800:04d}.png").convert("RGB").resize((w, h))
        # mask_image = np.array(mask_image) / 255.
        # mask_image = torch.from_numpy(mask_image).unsqueeze(0).unsqueeze(0).float().cuda()
        # image_fuse = steg_decode * (1 - mask_image) + 1 * mask_image

        # image_fuse = steg_decode

        #################
        #   backward:   #
        #################
        output_steg = dwt(image_fuse)
        
        output_image, bits = net(output_steg, rev=True)
        secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
        secret_rev = iwt(secret_rev)

        resi_cover = (steg_img - cover) * 20
        resi_secret = (secret_rev - secret) * 20

        secret_rev1 = secret_rev.cpu().numpy().squeeze() * 255
        np.clip(secret_rev1, 0, 255)
        secret1 = secret.cpu().numpy().squeeze() * 255
        np.clip(secret1, 0, 255)
        cover1 = cover.cpu().numpy().squeeze() * 255
        np.clip(cover1, 0, 255)
        # steg = image_fuse.cpu().numpy().squeeze() * 255
        steg = steg_img.cpu().numpy().squeeze() * 255
        np.clip(steg, 0, 255)
       
        psnr_temp = computePSNR(secret_rev1, secret1)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(cover1, steg)
        psnr_c.append(psnr_temp_c)

        with open('psnr_values6.txt', 'a') as file:
            file.write(f'PSNR: {str(psnr_temp_c)}\n')

        print("PSNR 已保存到 psnr_values.txt 文件中")

        acc = 1 - calculate_ber(bits, message)
        bit_acc.append(acc)
        print(acc)

        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
        torchvision.utils.save_image(out_temp, c.IMAGE_PATH_temp + '%.5d.png' % i)
        torchvision.utils.save_image(image_fuse, c.IMAGE_PATH_fuse + '%.5d.png' % i)
    
    print("PSNR_C", np.mean(psnr_c))
    print("PSNR_S", np.mean(psnr_s))
    print("Bit_ACC", np.mean(bit_acc))




