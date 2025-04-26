# %%
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # debug mode, will slower training
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import numpy as np
import accelerate
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm.scheduler
import kornia.augmentation as A
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from omniguard_cop import models, losses, datasets, transforms, utils


# %%
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# %% [markdown]
# # config

# %%
# train
parser = argparse.ArgumentParser()
parser.add_argument('--train-root', type=str, default='/dataset/MIRFlickR/train/')
parser.add_argument('--val-root', type=str, default='/dataset/MIRFlickR/val/')
parser.add_argument('--mask-root', type=str, default='/dataset/maskcocobig/')
parser.add_argument('--vae-root', type=str, default='CompVis/stable-diffusion-v1-4')
parser.add_argument('-n', '--num-epochs', type=int, default=150)
parser.add_argument('-b', '--batch-size', type=int, default=8)
parser.add_argument('-a', '--accumulate-steps', type=int, default=4)
parser.add_argument('-w', '--workers', type=int, default=32)
parser.add_argument('--pretrained', required=True, type=str)
parser.add_argument('--test-only', action='store_true', default=False)
args = parser.parse_args()

model_tag = 'finetune'
num_epochs = args.num_epochs
base_lr = 32 * 4e-6
base_batch_size = 32
batch_size = args.batch_size
log_freq = 100
num_workers = args.workers
accumulate_steps = args.accumulate_steps
save_dir = os.path.join('checkpoint', f"{now}_{model_tag}")
os.makedirs(save_dir, exist_ok=True)
pretrained = args.pretrained
crv_probs = [1, 1, 1]
final_ramp = 10000

# dataset
train_dir = args.train_root
val_dir = args.val_root
mask_dir = args.mask_root
vae_model = args.vae_root

# encoder
resolution = 256
secret_len = 100
# decoder
decoder_pretrained = True
resize = 224

# loss
pixel_loss = 'l2'
pixel_w = 1.5
frequency_w = 1.5
perceptual_w = 1.0
# loss_w settings
secret_w = 20   # start as image : secret = 1       : secret_w, default as 20
image_w = 27.5  #   end as image : secret = image_w : 1       , default as 27.5
save_freq = 1

# %%
# device = torch.device('cuda', index=0)
accelerator = accelerate.Accelerator(gradient_accumulation_steps=accumulate_steps)
device = accelerator.device
nproc = accelerator.num_processes

# %% [markdown]
# # dataset

# %%
train_transform = T.Compose([
    T.RandomResizedCrop(
        size=(resolution, resolution), scale=(0.8, 1.0), ratio=(3/4, 4/3), 
        interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0, 1] -> [-1, 1]
])

val_transform = T.Compose([
    T.CenterCrop(size=(resolution, resolution)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0, 1] -> [-1, 1]
])

train_set = datasets.ImageFolder(
    train_dir,
    file_level=2,
    secret_len=secret_len,
    transform=train_transform,
)

val_set = datasets.ImageFolder(
    val_dir,
    file_level=2,
    secret_len=secret_len,
    transform=val_transform,
)

# %%
train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
)
val_loader = DataLoader(
    val_set, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
)

# %%

perturbator = transforms.ImagePerturbator(
    base_perturb=A.AugmentationSequential(
        A.RandomHorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(resize, resize), scale=(0.8, 1.0), ratio=(3/4, 4/3), 
            resample='bilinear', 
            cropping_mode="resample", align_corners=False),  # trivial settings
    ),
    opt_perturbs=nn.ModuleList([
        A.RandomJPEG(jpeg_quality=torch.FloatTensor([40, 100]).to(device)),  # bug when inputting cuda tensor
        A.RandomBrightness(brightness=(0.5, 1.5)),
        A.RandomContrast(contrast=(0.5, 1.5)),
        A.ColorJiggle(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        A.RandomGrayscale(p=0.5),
        A.RandomGaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        A.RandomGaussianNoise(std=0.08),
        A.RandomHue(hue=0.05),
        A.RandomPosterize(bits=3),
        A.RandomRGBShift(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1),
        A.RandomSaturation(saturation=(0.5, 1.5)),
        A.RandomSharpness(sharpness=2.5),   
        A.RandomMedianBlur(kernel_size=3),
        A.RandomBoxBlur(kernel_size=7),
        A.RandomMotionBlur(kernel_size=(3, 9), angle=(-90.0, 90.0), direction=(-1.0, 1.0)),
    ]),
    # it seems that 2 or more sequential augmentations might become undifferentable
    opt_nprobs=[0.1, 0, 0.9],
)

perturbator_rwm = transforms.RandomWatermarkRemoval(mask_dir)
perturbator_vae = transforms.VAE(vae_model)
perturbator_vae = perturbator_vae.to(device)

# %% [markdown]
# # model

# %%
encoder = models.Encoder(
    resolution=resolution,
    secret_len=secret_len,
)

decoder = models.Decoder(
    secret_len=secret_len,
    pretrained=decoder_pretrained,
    resize=resize,
)

model = models.StegModel(encoder, decoder)
utils.count_params(model, True)
if pretrained:
    print('-'*32)
    print('loading pretrained weights...')
    print(model.load_state_dict(torch.load(pretrained), strict=True))

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = model.to(device)

# %%
calc_loss = losses.ImageSecretLoss(pixel_loss=pixel_loss)

# %%
actual_batch_size = batch_size * nproc * accumulate_steps
lr = np.sqrt(actual_batch_size / base_batch_size) * base_lr
optim = torch.optim.AdamW(model.parameters(), lr=lr)
sched = timm.scheduler.CosineLRScheduler(optim, t_initial=num_epochs)

# %%
model, calc_loss, optim, train_loader, val_loader = accelerator.prepare(
    model, calc_loss, optim, train_loader, val_loader)

# %% [markdown]
# # train

# %%
# # debug mode, will GREATLY slower training
# torch.autograd.set_detect_anomaly(True)
global_step = 0
cur_image_w, cur_secret_w = 0, 0
best_image_loss, best_bit_acc = float('inf'), 0.0

if args.test_only:
    model.eval()

    bit_acc_list, image_loss_list = [], []
    with torch.inference_mode():
        for idx, sample in tqdm(enumerate(val_loader)):
            inputs, secrets = sample['image'], sample['secret']
            
            inputs, secrets = inputs.to(torch.float32), secrets.to(torch.float32)
            inputs, secrets = inputs.to(device), secrets.to(device)
    
            # 1. encode
            reconstructions = model(inputs, secrets, mode='encode')
    
            # 2. perturbation
            perturb_w = 1.0
            
            crv_idx = torch.multinomial(
                torch.FloatTensor(crv_probs), num_samples=1).to(device)
            crv_idx = accelerate.utils.broadcast(crv_idx)
            if crv_idx == 0:
                opt_num = torch.LongTensor([perturbator.random_opt_num()]).to(device)
                opt_num = accelerate.utils.broadcast(opt_num).item()
                perturb_idxs = perturbator.random_perturb_idxs(opt_num).to(device)
                perturb_idxs = accelerate.utils.broadcast(perturb_idxs)
                
                reconstructions_perturbed = perturbator(reconstructions, 1.0, perturb_idxs)
            elif crv_idx == 1:
                mask_idx = torch.LongTensor([perturbator_rwm.random_mask_idx()]).to(device)
                mask_idx = accelerate.utils.broadcast(mask_idx).item()
                
                reconstructions_perturbed = perturbator_rwm(reconstructions, inputs, mask_idx)
                reconstructions_perturbed = (
                    perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)
            elif crv_idx == 2:
                reconstructions_perturbed = perturbator_vae(reconstructions)
                reconstructions_perturbed = (
                    perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)  
    
            # 3. decode
            secrets_logit = model(reconstructions_perturbed, mode='decode')
    
            cur_image_w = image_w * secret_w
            cur_secret_w = secret_w

            loss, loss_dict = calc_loss(
                inputs, reconstructions, cur_image_w,
                secrets, secrets_logit, cur_secret_w,
                pixel_w=pixel_w, perceptual_w=perceptual_w, frequency_w=frequency_w)
            
            secrets_pred = (secrets_logit > 0).to(torch.float32)
            all_secrets, all_secrets_pred = accelerator.gather_for_metrics((secrets, secrets_pred))
            all_bit_acc = torch.mean((all_secrets_pred == all_secrets).to(torch.float32))

            image_loss = torch.FloatTensor([loss_dict['image_loss']]).to(accelerator.device)
            all_image_loss = accelerator.gather_for_metrics(image_loss)
            all_image_loss = torch.mean(all_image_loss)

            bit_acc_list.append(all_bit_acc.item())
            image_loss_list.append(all_image_loss.item())
    
    bit_acc, image_loss = np.mean(bit_acc_list), np.mean(image_loss_list)
    if accelerator.is_main_process:
        print(f"[val]"
                f" bit_acc: {bit_acc:.3f}"
                f" image_loss: {image_loss:.3f}")

else:
    for epoch in range(num_epochs):
        model.train()

        sched.step(epoch)
        print(f"[epoch {epoch:03d}] lr: {optim.param_groups[0]['lr']}")
        bit_acc_list, image_loss_list = [], []
        for idx, sample in tqdm(enumerate(train_loader)):
            with accelerator.accumulate(model):
                inputs, secrets = sample['image'], sample['secret']
                
                inputs, secrets = inputs.to(torch.float32), secrets.to(torch.float32)
                inputs, secrets = inputs.to(device), secrets.to(device)
        
                # 1. encode
                reconstructions = model(inputs, secrets, mode='encode')
        
                # 2. perturbation
                perturb_w = min(global_step / final_ramp , 1.0)
                
                crv_idx = torch.multinomial(
                    torch.FloatTensor(crv_probs), num_samples=1).to(device)
                crv_idx = accelerate.utils.broadcast(crv_idx)
                if crv_idx == 0:
                    opt_num = torch.LongTensor([perturbator.random_opt_num()]).to(device)
                    opt_num = accelerate.utils.broadcast(opt_num).item()
                    perturb_idxs = perturbator.random_perturb_idxs(opt_num).to(device)
                    perturb_idxs = accelerate.utils.broadcast(perturb_idxs)
                    
                    reconstructions_perturbed = perturbator(reconstructions, 1.0, perturb_idxs)
                elif crv_idx == 1:
                    mask_idx = torch.LongTensor([perturbator_rwm.random_mask_idx()]).to(device)
                    mask_idx = accelerate.utils.broadcast(mask_idx).item()
                    
                    reconstructions_perturbed = perturbator_rwm(reconstructions, inputs, mask_idx)
                    reconstructions_perturbed = (
                        perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)
                elif crv_idx == 2:
                    reconstructions_perturbed = perturbator_vae(reconstructions)
                    reconstructions_perturbed = (
                        perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)  
        
                # 3. decode
                secrets_logit = model(reconstructions_perturbed, mode='decode')
        
                # 4. backward
                cur_image_w = image_w * secret_w
                cur_secret_w = secret_w

                loss, loss_dict = calc_loss(
                    inputs, reconstructions, cur_image_w,
                    secrets, secrets_logit, cur_secret_w,
                    pixel_w=pixel_w, perceptual_w=perceptual_w, frequency_w=frequency_w)
        
                optim.zero_grad()
                accelerator.backward(loss)
                find_nan = False
                for name, param in model.named_parameters():
                    if param.grad.isnan().any():
                        find_nan = True
                        param.grad.nan_to_num_()
                if find_nan: 
                    print('-' * 32)
                    print('find NaN gradients, replace it with 0.')
                optim.step()
                
                secrets_pred = (secrets_logit > 0).to(torch.float32)
                all_secrets, all_secrets_pred = accelerator.gather_for_metrics((secrets, secrets_pred))
                all_bit_acc = torch.mean((all_secrets_pred == all_secrets).to(torch.float32))

                image_loss = torch.FloatTensor([loss_dict['image_loss']]).to(accelerator.device)
                all_image_loss = accelerator.gather_for_metrics(image_loss)
                all_image_loss = torch.mean(all_image_loss)

                bit_acc_list.append(all_bit_acc.item())
                image_loss_list.append(all_image_loss.item())
        
                if (global_step + 1) % log_freq == 0:
                    if accelerator.is_main_process:
                        print('-'*32)
                        print(f"[epoch {epoch:03d}, step {idx:03d}] "
                            f" bit_acc: {all_bit_acc:.3f}"
                            f" image_loss: {all_image_loss:.3f}")
                        print('-'*32)
                        loss_dict['perturb_w'] = perturb_w
                        loss_dict['lr'] = optim.param_groups[0]['lr']
                        print(loss_dict)

                global_step += 1

        bit_acc, image_loss = np.mean(bit_acc_list), np.mean(image_loss_list)
        if accelerator.is_main_process:
            print(f"[epoch {epoch:03d} train]"
                f" bit_acc: {bit_acc:.3f}"
                f" image_loss: {image_loss:.3f}")
                
        if (epoch + 1) % save_freq == 0:
            model.eval()

            bit_acc_list, image_loss_list = [], []
            with torch.inference_mode():
                for idx, sample in tqdm(enumerate(val_loader)):
                    inputs, secrets = sample['image'], sample['secret']
                    
                    inputs, secrets = inputs.to(torch.float32), secrets.to(torch.float32)
                    inputs, secrets = inputs.to(device), secrets.to(device)
            
                    # 1. encode
                    reconstructions = model(inputs, secrets, mode='encode')
            
                    # 2. perturbation
                    perturb_w = 1.0
                    
                    crv_idx = torch.multinomial(
                        torch.FloatTensor(crv_probs), num_samples=1).to(device)
                    crv_idx = accelerate.utils.broadcast(crv_idx)
                    if crv_idx == 0:
                        opt_num = torch.LongTensor([perturbator.random_opt_num()]).to(device)
                        opt_num = accelerate.utils.broadcast(opt_num).item()
                        perturb_idxs = perturbator.random_perturb_idxs(opt_num).to(device)
                        perturb_idxs = accelerate.utils.broadcast(perturb_idxs)
                        
                        reconstructions_perturbed = perturbator(reconstructions, 1.0, perturb_idxs)
                    elif crv_idx == 1:
                        mask_idx = torch.LongTensor([perturbator_rwm.random_mask_idx()]).to(device)
                        mask_idx = accelerate.utils.broadcast(mask_idx).item()
                        
                        reconstructions_perturbed = perturbator_rwm(reconstructions, inputs, mask_idx)
                        reconstructions_perturbed = (
                            perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)
                    elif crv_idx == 2:
                        reconstructions_perturbed = perturbator_vae(reconstructions)
                        reconstructions_perturbed = (
                            perturb_w * reconstructions_perturbed + (1 - perturb_w) * reconstructions)  
            
                    # 3. decode
                    secrets_logit = model(reconstructions_perturbed, mode='decode')
            
                    cur_image_w = image_w * secret_w
                    cur_secret_w = secret_w
        
                    loss, loss_dict = calc_loss(
                        inputs, reconstructions, cur_image_w,
                        secrets, secrets_logit, cur_secret_w,
                        pixel_w=pixel_w, perceptual_w=perceptual_w, frequency_w=frequency_w)
                    
                    secrets_pred = (secrets_logit > 0).to(torch.float32)
                    all_secrets, all_secrets_pred = accelerator.gather_for_metrics((secrets, secrets_pred))
                    all_bit_acc = torch.mean((all_secrets_pred == all_secrets).to(torch.float32))

                    image_loss = torch.FloatTensor([loss_dict['image_loss']]).to(accelerator.device)
                    all_image_loss = accelerator.gather_for_metrics(image_loss)
                    all_image_loss = torch.mean(all_image_loss)

                    bit_acc_list.append(all_bit_acc.item())
                    image_loss_list.append(all_image_loss.item())
            
            bit_acc, image_loss = np.mean(bit_acc_list), np.mean(image_loss_list)
            if accelerator.is_main_process:
                print(f"[epoch {epoch:03d} val]"
                    f" bit_acc: {bit_acc:.3f}"
                    f" image_loss: {image_loss:.3f}")
            
            save_model = False
            if bit_acc > best_bit_acc:
                best_bit_acc = bit_acc
                save_model = True    
            if image_loss < best_image_loss:
                best_image_loss = image_loss
                save_model = True
            if accelerator.is_main_process and save_model:
                torch.save(
                    model.module.state_dict(), 
                    os.path.join(save_dir, f"omniguard-cop_epoch{epoch:03d}"
                                        f"_bit-acc{bit_acc:.3f}"
                                        f"_image-loss{image_loss:.4f}.pth"))


