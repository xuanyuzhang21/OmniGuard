import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 
import torchvision.models as models


# class DoubleConv(nn.Module):
#     """(convolution => BN => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
        
#         print(f"diffY: {diffY} | diffX: {diffX}")

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    
    
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
            self.use_bias = False
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':   
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x)) 
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):   
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'




class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x




class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, activ='relu', norm='none'):
        super().__init__()
        # initialize activation
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        elif activ == 'selu':
            self.activ = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'silu':
            self.activ = nn.SiLU(inplace=True)
        elif activ == 'none':
            self.activ = None
        else:
            assert 0, "Unsupported activ: {}".format(activ)

        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
            bias = False
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels='default', out_channels='default', activ='relu', norm='none'):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels//2
        if out_channels == 'default':
            out_channels = in_channels//2
        self.up = nn.Upsample(scale_factor=(2,2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)

        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)

    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class DecBlockBilinear(nn.Module):
    def __init__(self, in_channels, skip_channels='default', out_channels='default', activ='relu', norm='none'):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels//2
        if out_channels == 'default':
            out_channels = in_channels//2
        self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)

        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)

    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class Secret2Image(nn.Module):
    def __init__(self, resolution, secret_len):
        super().__init__()
        assert resolution % 16 == 0, "Resolution must be a multiple of 16."
        self.dense = nn.Linear(secret_len, 16*16*3)
        self.upsample = nn.Upsample(
            scale_factor=(resolution//16, resolution//16), mode='nearest')
        self.activ = nn.ReLU(inplace=True)

    def forward(self, secret):
        x = self.dense(secret)
        x = x.view((-1, 3, 16, 16))
        x = self.activ(self.upsample(x))
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 resolution=256, 
                 secret_len=100,
                 width=32, 
                 ndown=4, 
                 nmiddle=1, 
                 activ='silu'):
        super().__init__()
        self.secret_len = secret_len
        self.ndown = ndown
        self.resolution = resolution
        self.secret2image = Secret2Image(resolution, secret_len)
        self.pre = Conv2d(6, width, 3, 1, 1, activ=activ, norm='none')
        
        self.enc = nn.ModuleList()
        ch = width
        for _ in range(ndown):
            self.enc.append(Conv2d(ch, ch*2, 3, 2, 1, norm='none'))
            ch *= 2

        if nmiddle > 0:
            self.middle = nn.ModuleList()
            for _ in range(nmiddle):
                self.middle.append(ResBlocks(nmiddle, ch, norm='none', activation=activ, pad_type='zero'))
        else:
            self.middle = None

        self.dec = nn.ModuleList()
        for i in range(ndown):
            skip_width = ch//2 if i < ndown-1 else ch//2+6  # 6 is in_chs
            self.dec.append(DecBlock(ch, skip_width, activ=activ, norm='none'))
            ch //= 2

        # TODO is tanh really necessary?
        self.post = nn.Sequential(
            Conv2d(ch, ch, 3, 1, 1, activ=activ, norm='none'),
            Conv2d(ch, ch//2, 1, 1, 0, activ='silu', norm='none'),
            Conv2d(ch//2, 3, 1, 1, 0, activ='tanh', norm='none')
        )

    def forward(self, image, secret=None):
        if secret is None: 
            secret = torch.randn(image.shape[0], self.secret_len, device=image.device)
            
        secret = self.secret2image(secret)
        inputs = torch.cat([image, secret], dim=1)
        enc = []
        x = self.pre(inputs)
        for block in self.enc:
            enc.append(x)
            x = block(x)
        enc = enc[::-1]
        if self.middle:
            for block in self.middle:
                x = block(x)
        for i, (block, skip) in enumerate(zip(self.dec, enc)):
            if i < self.ndown-1:
                x = block(x, skip)
            else:
                x = block(x, torch.cat([skip, inputs], dim=1))
        x = self.post(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        secret_len,
        pretrained=True,
        resize=224,
    ):
        super().__init__()
        if resize is not None:
            self.resize = T.Resize(size=(resize, resize),
                                   interpolation=T.InterpolationMode.BILINEAR)
        else:
            self.resize = nn.Identity()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, secret_len)
    
    def forward(self, x):
        x = self.resize(x)
        x = self.backbone(x)
        return x
    
    
class StegModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, bit=None, mode='encode'):
        assert mode in ['encode', 'decode']
        if mode == 'encode':
            return self.encoder(x, bit)
        if mode == 'decode':
            return self.decoder(x)