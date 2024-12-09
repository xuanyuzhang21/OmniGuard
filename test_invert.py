import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model_invert import *
import config as c
import datasets
import modules.Unet_common as common
import random
from Quantization import Quantization
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from JPEG import DiffJPEG
from jpegtest import JpegTest
import kornia
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

class GaussianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), sigma=(2,2), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask
        return self.transform(image)

class MedianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, image_cover_mask):
        image = image_cover_mask
        return self.transform(image)

class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, image_cover_mask):
        image = image_cover_mask
        #
        noised_down = F.interpolate(
                                    image,
                                    size=(int(self.down_scale * image.shape[2]), int(self.down_scale * image.shape[3])),
                                    mode='nearest'
                                    )
        noised_up = F.interpolate(
                                    noised_down,
                                    size=(image.shape[2], image.shape[3]),
                                    mode='nearest'
                                    )

        return noised_up

class Brightness(nn.Module):
    def __init__(self, brightness=0.5, p=1):
        super(Brightness, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(brightness=brightness, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        colorjitter = self.transform(image)
        return colorjitter

class Contrast(nn.Module):
    def __init__(self, contrast=0.5, p=1):
        super(Contrast, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(contrast=contrast, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = image
        colorjitter = self.transform(out)
        return colorjitter

class Hue(nn.Module):

    def __init__(self, hue=0.1, p=1):
        super(Hue, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(hue=hue, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask
        colorjitter = self.transform(image)
        return colorjitter

class SaltPepper(nn.Module):

	def __init__(self, prob=0.01):
		super(SaltPepper, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2., prob / 2.])).to(image.device)
		mask = mask.expand_as(image)

		image[mask == 1] = 1  # salt
		image[mask == 2] = 0  # pepper

		return image

	def forward(self, image_cover_mask):
		image = image_cover_mask
		return self.sp_noise(image, self.prob)

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

device = torch.device(c.device_ids[0] if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    for k, v in state_dicts['net'].items():
        print(k)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if ('tmp_var' not in k) and ('bm' not in k)}
    net.load_state_dict(network_state_dict, strict=False)
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
net.to(device)
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()


# TM_SCHEMA_CODE=TrustMark.Encoding.BCH_5
# tm=TrustMark(verbose=True, model_type='Q', encoding_type=TM_SCHEMA_CODE)
# bitlen=tm.schemaCapacity()
# print(bitlen)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "/gdata/cold1/zhangxuanyu/sd_inpaint",
                torch_dtype=torch.float16,
            ).to(device)

# from diffusers import StableDiffusionInstructPix2PixPipeline
# # 加载 InstructPix2Pix 模型
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")
# pipe = pipe.to("cuda")

# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32
# ).to("cuda")
# pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "/data/zhangxuanyu/sd15/snapshots/39593d5650112b4cc580433f6b0435385882d819", controlnet=controlnet, torch_dtype=torch.float32
# ).to("cuda")

with torch.no_grad():
    psnr_s = []
    psnr_c = []
    bit_acc = []
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        # cover = data[data.shape[0] // 2:, :, :, :]
        cover = data

        b, _, H, W = cover.shape

        R = 0
        G = 0
        B = 255
        # image = Image.new('RGB', (W, H), (R, G, B))
        image = Image.open("/gdata/cold1/zhangxuanyu/HiNet/bluesky_white2.png").convert("RGB").resize((W, H))
        # image = Image.open("/data/zhangxuanyu/HiNet/bluesky.jpg").convert("RGB").resize((W, H))
        # image = Image.open("/data/zhangxuanyu/HiNet/blueskyyy.png").convert("RGB").resize((W, H))
        result = np.array(image) / 255.
        expanded_matrix = np.expand_dims(result, axis=0) 
        expanded_matrix = np.repeat(expanded_matrix, b, axis=0)
        secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
        secret = secret.permute(0, 3, 1, 2).to(device)

        # secret = cover

        cover_input = dwt(cover)
        secret_input = dwt(secret)
        message = torch.randint(2, (1, 64)).to(torch.float32).to(device)

        steg_img, output_z, out_temp, secret_temp = net(cover_input, secret_input, message)

        quant = Quantization()

        # image_fuse = cover

        # steg_img = image_fuse

        # image_fuse = steg_img

        # import random
        # choice = random.randint(0, 6)
        # choice = 3
        # choice = 7

        b, _, h, w = steg_img.shape

        # masksrc = "/data/zhangxuanyu/segmentanything/maskcocobig/"
        # masksrc = "/gdata/cold1/zhangxuanyu/segmentanything/cocomaskfinal/"
        masksrc = "/gdata/cold1/zhangxuanyu/HiNet/"
        mask_image = Image.open(masksrc + str(3).zfill(4) + "_mask.png").convert("L")
        mask_image = mask_image.resize((w, h))
        h, w = mask_image.size
        
        image = steg_img.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
        prompt = ""
        image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
        image_inpaint = pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
        image_inpaint = np.array(image_inpaint) / 255.
        mask_image = np.array(mask_image)
        mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
        mask_image = mask_image.astype(np.uint8)
        image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
        # image_fuse = image_inpaint
        image_fuse = torch.from_numpy(image_fuse).permute(2,0,1).unsqueeze(0).float().to(device)
            
        # if choice == 0:
        #     # NL = float((np.random.randint(1, 10))/255)
        #     NL = float(10 / 255)
        #     noise = np.random.normal(0, NL, steg_img.shape)
        #     torchnoise = torch.from_numpy(noise).to(device).float()

        #     steg_img_tmp = image_fuse.clone() + torchnoise
        #     image_fuse = quant(steg_img_tmp)
        
        # elif choice == 1:
        #     NL = 70
        #     # NL = int(np.random.randint(70, 85))
        #     DiffJPEG_opt = JpegTest(NL, 0)
        #     image_fuse = DiffJPEG_opt(image_fuse)

        # elif choice == 2:
        #     image_fuse = quant(image_fuse)

        # elif choice == 3:
        #     bright = Brightness()
        #     steg_img_tmp = bright(image_fuse)
        #     image_fuse = quant(steg_img_tmp)

        # elif choice == 4:
        #     contrast = Contrast()
        #     steg_img_tmp = contrast(image_fuse)
        #     image_fuse = quant(steg_img_tmp)
        
        # elif choice == 5:
        #     hue_opt = Hue()
        #     steg_img_tmp = hue_opt(image_fuse)
        #     image_fuse = quant(steg_img_tmp)

        # elif choice == 6:
        #     sp = SaltPepper()
        #     steg_img_tmp = sp(image_fuse)
        #     image_fuse = quant(steg_img_tmp)  

        import random
        from PIL import Image
        import os

        # b, _, h, w = steg_decode.shape

        # # masksrc = "/data/zhangxuanyu/segmentanything/maskcocobig/"
        # masksrc = "/data/zhangxuanyu/segmentanything/cocomaskfinal/"
        # # masksrc = "/data/zhangxuanyu/HiNet/"
        # mask_image = Image.open(masksrc + str(i+1).zfill(4) + ".png").convert("L")
        # mask_image = mask_image.resize((w, h))
        # h, w = mask_image.size
        
        # image = steg_decode.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
        # prompt = ""
        # image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
        # image_inpaint = pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
        # image_inpaint = np.array(image_inpaint) / 255.
        # mask_image = np.array(mask_image)
        # mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
        # mask_image = mask_image.astype(np.uint8)
        # image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
        # # image_fuse = image_inpaint
        # image_fuse = torch.from_numpy(image_fuse).permute(2,0,1).unsqueeze(0).float().to(device)

        # generator = torch.Generator(device="cuda").manual_seed(1)
        # mask_path = "/data/zhangxuanyu/segmentanything/cocomaskfinal/" + str(i+1).zfill(4) + ".png"
        # mask_image = load_image(mask_path)
        # mask_image = mask_image.resize((512, 512))
        # image_init = steg_decode.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
        # image_init1 = Image.fromarray((image_init * 255).astype(np.uint8), mode = "RGB")
        # image_mask = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        # assert image_init.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        # image_init[image_mask > 0.5] = -1.0  # set as masked pixel
        # image = np.expand_dims(image_init, 0).transpose(0, 3, 1, 2)
        # control_image = torch.from_numpy(image)

        # # generate image
        # image_inpaint = pipe_control(
        #     "",
        #     num_inference_steps=20,
        #     generator=generator,
        #     eta=1.0,
        #     image=image_init1,
        #     mask_image=image_mask,
        #     control_image=control_image,
        # ).images[0]
        
        # image_inpaint = np.array(image_inpaint) / 255.
        # image_mask = np.stack([image_mask] * 3, axis=-1)
        # image_mask = image_mask.astype(np.uint8)
        # image_fuse = image_init * (1 - image_mask) + image_inpaint * image_mask
        # image_fuse = torch.from_numpy(image_fuse).permute(2,0,1).unsqueeze(0).float().to(device)

        #################
        #   backward:   #
        #################
        output_steg = dwt(image_fuse)
        
        output_image, bits = net(output_steg.to(device), rev=True)
        secret_rev = output_image.narrow(1, 0, 12)
        secret_rev = iwt(secret_rev)

        steg_img = steg_img.to(device)
        secret_rev = secret_rev.to(device)
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
        print(i, psnr_temp_c)
        psnr_c.append(psnr_temp_c)

        with open('psnr_values6.txt', 'a') as file:
            file.write(f'PSNR: {str(psnr_temp_c)}\n')

        print("PSNR 已保存到 psnr_values.txt 文件中")

        # message = str(message.tolist())

        bits = (bits > 0).int()

        message = (message > 0).int()

        # 将结果转换为 Python 列表，然后连接成一个字符串
        binary_string = ''.join(str(x.item()) for x in bits.flatten())
        print(binary_string)
        message = ''.join(str(x.item()) for x in message.flatten())
        print(message)

        acc = 1 - calculate_ber(binary_string, message)
        bit_acc.append(acc)
        print(acc)

        # secret = iwt(secret_temp)

        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        # torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
        # torchvision.utils.save_image(out_temp, c.IMAGE_PATH_temp + '%.5d.png' % i)
        torchvision.utils.save_image(image_fuse, c.IMAGE_PATH_fuse + '%.5d.png' % i)
    
    print("PSNR_C", np.mean(psnr_c))
    print("PSNR_S", np.mean(psnr_s))
    print("Bit_ACC", np.mean(bit_acc))




