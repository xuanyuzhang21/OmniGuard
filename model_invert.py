import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet, InvertibleTransform
# from trustmark import TrustMark
from unet import Unet1, SecretDecoder
import torch.nn.functional as F
import kornia
from collections import OrderedDict
# TM_SCHEMA_CODE=TrustMark.Encoding.BCH_4
# TM_SCHEMA_CODE=TrustMark.Encoding.BCH_5

import modules.Unet_common as common
dwt = common.DWT()
iwt = common.IWT()

device = torch.device(c.device_ids[0] if torch.cuda.is_available() else "cpu")

class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64, model='MIMO-VRN'):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
        # but this is how we trained the model in the first place and what we reported in the paper
        if model == 'LSTM-VRN':
            self.relu = nn.ReLU(inplace=True)
        elif model == 'MIMO-VRN':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class PredictiveModuleMIMO(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=8):
        super(PredictiveModuleMIMO, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        x = self.conv_in(x)
        res = self.res_block(x) + x

        return res

##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=12,prompt_len=3,prompt_size = 36,lin_dim = 12):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class PredictiveModuleMIMO_prompt(nn.Module):
    def __init__(self, channel_in, nf, prompt_len=2, block_num_rbm=8):
        super(PredictiveModuleMIMO_prompt, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        trans_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))

        self.res_block = nn.Sequential(*res_block)
        self.prompt = PromptGenBlock(prompt_dim=nf,prompt_len=prompt_len,prompt_size = 36,lin_dim = nf)
        self.fuse = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv_in(x)
        res = self.res_block(x)
        prompt = self.prompt(res)

        result = self.fuse(torch.cat([res, prompt], dim=1))

        return result

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class BitModule(nn.Module):
    def __init__(self):
        super(BitModule, self).__init__()
        self.encoder = Unet1()
        self.decoder = SecretDecoder()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()
        self.invert = InvertibleTransform()
        self.bm = BitModule()

        ck1 = torch.load('checkpoint/encoder_Q.ckpt')

        new_state_dict = OrderedDict()
        for k, v in ck1.items():
            if k.startswith("encoder."):
                new_key = k[len("encoder."):]  # 去掉 'encoder.' 前缀
            else:
                new_key = k
            new_state_dict[new_key] = v
        self.bm.encoder.load_state_dict(new_state_dict, strict=True)

        ck2 = torch.load('checkpoint/decoder_Q.ckpt')

        new_state_dict1 = OrderedDict()
        for k, v in ck2.items():
            # 检查是否有多余的 'decoder.decoder' 前缀
            if k.startswith("decoder.decoder"):
                new_key = k.replace("decoder.decoder", "decoder")  # 将 'decoder.decoder' 替换为 'decoder'
            else:
                new_key = k
            new_state_dict1[new_key] = v

        self.bm.decoder.load_state_dict(new_state_dict1, strict=True)

        self.decodenet = PredictiveModuleMIMO_prompt(12, 36)

        # 冻结 self.bm 中的所有参数
        for param in self.bm.parameters():
            param.requires_grad = False

    def encode(self, trustmark1, cover, secret, MODE='text', WM_STRENGTH=1.0, WM_MERGE='BILINEAR'): 
        b, c, h, w = cover.shape

        # align_corners=False consistent with PIL
        transform = kornia.augmentation.Resize((256, 256), resample=WM_MERGE, align_corners=False)  
        # transform = kornia.augmentation.Resize((512, 512), resample=WM_MERGE, align_corners=False)
        transform_inv = kornia.augmentation.Resize((h, w), resample=WM_MERGE, align_corners=False)

        x = transform(cover)
        stego = trustmark1.encoder(x * 2 - 1, secret)
        stego = stego / 2 + 0.5
        residual = stego - x
        residual = transform_inv(residual)

        stego = cover + WM_STRENGTH * residual
        
        return stego

    def forward(self, cover_input, secret_input=None, watermarkID=None, rev=False):

        if not rev:
            concat = torch.cat((secret_input, cover_input), 1)
            secret_temp = self.invert(concat).narrow(1, 0, 4 * 3)
            x = torch.cat((cover_input, secret_temp), 1)
            output = self.model(x)
            output_steg = output.narrow(1, 0, 4 * 3)
            output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
            out = iwt(output_steg)
            out_temp = out
            
            # out = self.encode(self.bm, out.to(device), watermarkID)

            return out, output_z, out_temp, secret_temp

        else:
            x = cover_input
            x_gauss = self.decodenet(x)
            x_cat = torch.cat((x, x_gauss), 1)
            out = self.model(x_cat, rev=True)
            out_tmp = out.narrow(1, 12, out.shape[1] - 12)
            input_tmp = torch.cat((out_tmp, x), 1)
            res = self.invert(input_tmp)

            # out_freq = x.narrow(1, 0, 4 * 3)
            # img = iwt(out_freq)
            # img = img.to(device)
            
            # extracted_bits = self.bm.decoder(img * 2 - 1)
            
            return res


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
