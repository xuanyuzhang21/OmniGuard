# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
# pretraining

torchrun --nproc_per_node=8  train.py \
    --local_rank 0 --debug_slurm --output_dir <PRETRAINING_OUTPUT_DIRECTORY> \
    --augmentation_config configs/all_augs.yaml --extractor_model sam_base --embedder_model vae_small \
    --img_size 256 --batch_size 16 --batch_size_eval 32 --epochs 300 \
    --optimizer "AdamW,lr=5e-5" --scheduler "CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=10" \
    --seed 42 --perceptual_loss none --lambda_i 0.0 --lambda_d 0.0 --lambda_w 1.0 --lambda_w2 10.0 \
    --nbits 32 --scaling_i 1.0 --scaling_w 0.3 \
    --train_dir <COCO_TRAIN_DIRECTORY_PATH> --train_annotation_file <COCO_TRAIN_ANNOTATION_FILE_PATH> \
    --val_dir <COCO_VALIDATION_DIRECTORY_PATH> --val_annotation_file <COCO_VALIDATION_ANNOTATION_FILE_PATH> 

# finetuning

torchrun --nproc_per_node=8 train.py \
    --local_rank 0 --debug_slurm --output_dir <FINETUNING_OUTPUT_DIRECTORY> \
    --augmentation_config configs/all_augs_multi_wm.yaml --extractor_model sam_base --embedder_model vae_small \
    --resume_from <PRETRAINING_OUTPUT_DIRECTORY>/checkpoint.pth \
    --attenuation jnd_1_3_blue --img_size 256 --batch_size 8 --batch_size_eval 16 --epochs 200 \
    --optimizer "AdamW,lr=1e-4" --scheduler "CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5" \
    --seed 42 --perceptual_loss none --lambda_i 0 --lambda_d 0 --lambda_w 1.0 --lambda_w2 6.0 \
    --nbits 32 --scaling_i 1.0 --scaling_w 2.0 --multiple_w 1 --roll_probability 0.2 \
    --train_dir <COCO_TRAIN_DIRECTORY_PATH> --train_annotation_file <COCO_TRAIN_ANNOTATION_FILE_PATH> \
    --val_dir <COCO_VALIDATION_DIRECTORY_PATH> --val_annotation_file <COCO_VALIDATION_ANNOTATION_FILE_PATH>

# Or for evaluation only on 1 GPU:

python  train.py  --debug_slurm  --resume_from <FINETUNING_OUTPUT_DIRECTORY>/checkpoint.pth --augmentation_config configs/all_augs.yaml --extractor_model sam_base --embedder_model vae_small     --img_size 256 --batch_size 16 --batch_size_eval 32 --epochs 300     --optimizer "AdamW,lr=5e-5" --scheduler "CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=10"     --seed 42 --perceptual_loss none --lambda_i 0.0 --lambda_d 0.0 --lambda_w 1.0 --lambda_w2 10.0     --nbits 32 --scaling_i 1.0 --scaling_w 0.3     --output_dir output/ --debug_slurm --train_dir /datasets01/COCO-Stuff/042623/face_blurred/train_img/ --train_annotation_file /datasets01/COCO-Stuff/072318/stuff_train2017.json --val_dir /datasets01/COCO-Stuff/042623/face_blurred/val_img/ --val_annotation_file  "/datasets01/COCO-Stuff/072318/stuff_val2017.json" --only_eval True --nb_images_eval 50 --train_dir <COCO_TRAIN_DIRECTORY_PATH> --train_annotation_file <COCO_TRAIN_ANNOTATION_FILE_PATH> --val_dir <COCO_VALIDATION_DIRECTORY_PATH> --val_annotation_file <COCO_VALIDATION_ANNOTATION_FILE_PATH>
"""

import sys
import time
import datetime
import json
import os
import argparse
import random
import omegaconf
from typing import List

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CocoDetection

from watermark_anything.models import Wam, build_embedder, build_extractor
from watermark_anything.augmentation.augmenter import Augmenter
from watermark_anything.augmentation.geometric import Rotate, Resize, Crop, Perspective, HorizontalFlip, Identity, Combine, UpperLeftCrop, CropResizePad
from watermark_anything.augmentation.valuemetric import JPEG, GaussianBlur, MedianFilter, Brightness, Contrast, Saturation, Hue
from watermark_anything.data.transforms import get_transforms, get_transforms_segmentation, normalize_img, unnormalize_img, unstd_img
from watermark_anything.data.loader import get_dataloader, get_dataloader_segmentation
from watermark_anything.data.metrics import accuracy, psnr, iou, bit_accuracy, bit_accuracy_inference
from watermark_anything.losses.detperceptual import LPIPSWithDiscriminator
from watermark_anything.modules.jnd import JND
from watermark_anything.utils.image import create_diff_img, detect_wm_hm, create_fixed_color_palette, masks_to_colored_image

import watermark_anything.utils as utils
import watermark_anything.utils.dist as udist
import watermark_anything.utils.optim as uoptim
import watermark_anything.utils.logger as ulogger

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str,required=True)
    aa("--train_annotation_file", type=str, required=True)
    aa("--val_dir", type=str, required=True)
    aa("--val_annotation_file", type=str, required=True)
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Config paths')
    aa("--embedder_config", type=str, default="configs/embedder.yaml", help="Path to the embedder config file")
    aa("--augmentation_config", type=str, default="configs/all_augs.yaml", help="Path to the augmentation config file")
    aa("--extractor_config", type=str, default="configs/extractor.yaml", help="Path to the extractor config file")
    aa("--attenuation_config", type=str, default="configs/attenuation.yaml", help="Path to the attenuation config file")
    aa("--embedder_model", type=str, default=None, help="Name of the embedder model")
    aa("--extractor_model", type=str, default=None, help="Name of the extractor model")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--nbits", type=int, default=16, help="Number of bits used to generate the message. If 0, no message is used.")
    aa("--img_size", type=int, default=256, help="Size of the input images")
    aa("--img_size_extractor", type=int, default=256, help="Size of the input images")
    aa("--attenuation", type=str, default="None", help="Attenuation model to use")
    aa("--scaling_w", type=float, default=0.4, help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_i", type=float, default=1.0, help="Scaling factor for the image in the embedder model")

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=1e-4", help="Optimizer (default: AdamW,lr=1e-4)")
    aa("--optimizer_d", type=str, default=None, help="Discriminator optimizer. If None uses the same params (default: None)")
    aa("--scheduler", type=str, default= "None", help="Scheduler (default: None)")
    aa('--epochs', default=100, type=int, help='Number of total epochs to run')
    aa('--batch_size', default=16, type=int, help='Batch size')
    aa('--batch_size_eval', default=64, type=int, help='Batch size for evaluation')
    aa('--temperature', default=1.0, type=float, help='Temperature for the mask loss')
    aa('--workers', default=8, type=int, help='Number of data loading workers')
    aa('--resume_from', default=None, type=str, help='Path to the checkpoint to resume from')
    aa('--to_freeze_embedder', default=None, type=str, help='What parts of the embedder to freeze')

    group = parser.add_argument_group('Losses parameters')
    aa('--lambda_w', default=1.0, type=float, help='Weight for the watermark detection loss')
    aa('--lambda_w2', default=4.0, type=float, help='Weight for the watermark decoding loss')
    aa('--lambda_i', default=1.0, type=float, help='Weight for the image loss')
    aa('--lambda_d', default=0.5, type=float, help='Weight for the discriminator loss')
    aa('--balanced', type=utils.bool_inst, default=True, help='If True, the weights of the losses are balanced')
    aa('--total_gnorm', default=0.0, type=float, help='Total norm for the adaptive weights. If 0, uses the norm of the biggest weight.')
    aa('--perceptual_loss', default='lpips', type=str, help='Perceptual loss to use. "lpips", "watson_vgg" or "watson_fft"')
    aa('--disc_start', default=0, type=float, help='Weight for the discriminator loss')
    aa('--disc_num_layers', default=2, type=int, help='Number of layers for the discriminator')
    
    group = parser.add_argument_group('Misc.')
    aa('--only_eval', type=utils.bool_inst, default=False, help='If True, only runs evaluate')
    aa('--eval_freq', default=5, type=int, help='Frequency for evaluation')
    aa("--roll_probability", type=float, default=0, help="probability to inpaint betweem images of each batch.")
    aa("--multiple_w", type=float, default=0, help="probability to use 2 watermarks instead of 1.")
    aa("--eval_multiple_w", type=utils.bool_inst, default=False, help="evaluate with multiple watermarks.")
    aa("--nb_wm_eval", type=int, default=5, help="how many watermarks to use for evaluation (default: 5)")
    aa("--nb_images_eval", default=1000, type=int, help="Number of images to evaluate")
    aa("--nb_images_eval_multiple_w", default=1000, type=int, help="Number of images to evaluate for multiple wm. Takes longer.")
    aa('--saveimg_freq', default=5, type=int, help='Frequency for saving images')
    aa('--saveckpt_freq', default=50, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=0, type=int, help='Random seed')
    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def main(params):    
    """
    Main function to set up and run the training and evaluation of the watermarking model.
    This function handles distributed setup, model building, data loading, and the training loop.
    """

    # Initialize distributed mode if applicable
    udist.init_distributed_mode(params)
    if params.multiple_w > 0:
        print("Training with multiple watermarks. Adding multiple watermark evaluation.")
        params.eval_multiple_w = True

    # Set seeds for reproducibility
    seed = params.seed + udist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if params.distributed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Print the current git commit and parameters
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Copy configuration files to the output directory
    if udist.is_main_process():
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        os.system(f'cp {params.embedder_config} {params.output_dir}/configs/embedder.yaml')
        os.system(f'cp {params.augmentation_config} {params.output_dir}/configs/augs.yaml')
        os.system(f'cp {params.extractor_config} {params.output_dir}/configs/extractor.yaml')

    # Build the embedder model
    embedder_cfg = omegaconf.OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model, embedder_params, params.nbits)
    print(embedder)

    # Freeze specified parts of the embedder if needed
    if params.to_freeze_embedder is not None:
        to_freeze = params.to_freeze_embedder.split(',')
        if "encoder" in to_freeze:
            for param in embedder.encoder.parameters():
                param.requires_grad = False
        if "decoder" in to_freeze:
            for param in embedder.decoder.parameters():
                param.requires_grad = False
        if "msg_processor" in to_freeze:
            for param in embedder.msg_processor.parameters():
                param.requires_grad = False

    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Build the augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    augmenter = Augmenter(**augmenter_cfg)
    print(f'augmenter: {augmenter}')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_extractor, params.nbits)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Build attenuation if specified
    if params.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
        attenuation = JND(**attenuation_cfg[params.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
    else: 
        attenuation = None
    print(f'attenuation: {attenuation}')

    # Build the complete watermarking model
    wam = Wam(embedder, extractor, augmenter, attenuation, params.scaling_w, params.scaling_i, roll_probability=params.roll_probability) 
    wam.to(device)

    # Build the image detection loss
    image_detection_loss = LPIPSWithDiscriminator(
        balanced=params.balanced, total_norm=params.total_gnorm,
        disc_weight=params.lambda_d, percep_weight=params.lambda_i, 
        detect_weight=params.lambda_w, decode_weight=params.lambda_w2,
        disc_start=params.disc_start, disc_num_layers=params.disc_num_layers,
        percep_loss=params.perceptual_loss
    ).to(device)
    print(image_detection_loss)

    # Build optimizer and learning rate scheduler
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(
        model_params=list(embedder.parameters()) + list(extractor.parameters()), 
        **optim_params
    )
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(optimizer=optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # Build discriminator optimizer
    optim_params_d = uoptim.parse_params(params.optimizer) if params.optimizer_d is None else uoptim.parse_params(params.optimizer_d)
    optimizer_d = uoptim.build_optimizer(
        model_params=[*image_detection_loss.discriminator.parameters()], 
        **optim_params_d
    )
    print('optimizer_d: %s' % optimizer_d)

    # Data loaders for training and validation
    train_transform, train_mask_transform, val_transform, val_mask_transform = get_transforms_segmentation(params.img_size)

    if "COCO" in params.train_dir:
        train_loader = get_dataloader_segmentation(
            params.train_dir, params.train_annotation_file, 
            transform=train_transform, mask_transform=train_mask_transform,
            batch_size=params.batch_size, 
            num_workers=params.workers, shuffle=False, multi_w=params.multiple_w > 0
        )
        val_loader = get_dataloader_segmentation(
            params.val_dir, params.val_annotation_file, 
            transform=val_transform, mask_transform=val_mask_transform,
            batch_size=params.batch_size_eval, 
            num_workers=params.workers, shuffle=False, random_nb_object=False, multi_w=False
        )
        val_loader_multi_wm = get_dataloader_segmentation(
            params.val_dir, params.val_annotation_file, 
            transform=val_transform, mask_transform=val_mask_transform,
            batch_size=params.batch_size_eval, 
            num_workers=params.workers, shuffle=False, random_nb_object=False, multi_w=True, max_nb_masks=params.nb_wm_eval
        )
    else:
        train_loader = get_dataloader(
            params.train_dir, 
            transform=train_transform,
            batch_size=params.batch_size, 
            num_workers=params.workers, shuffle=True
        )
        val_loader = get_dataloader(
            params.val_dir,
            transform=val_transform,
            batch_size=params.batch_size_eval, 
            num_workers=params.workers, shuffle=False
        )
        val_loader_multi_wm = get_dataloader(
            params.val_dir,                                
            transform=val_transform,
            batch_size=params.batch_size_eval, 
            num_workers=params.workers, shuffle=False
        )

    # Optionally resume training from a checkpoint
    if params.resume_from is not None: 
        uoptim.restart_from_checkpoint(
            params.resume_from,
            model=wam,
        )
    to_restore = {"epoch": 0}
    uoptim.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=wam,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = optim_params_d['lr']
    optimizers = [optimizer, optimizer_d]

    # Setup for distributed training if applicable
    if params.distributed:
        wam_ddp = nn.parallel.DistributedDataParallel(wam, device_ids=[params.local_rank])
        image_detection_loss.discriminator = nn.parallel.DistributedDataParallel(
            image_detection_loss.discriminator, device_ids=[params.local_rank]
        )
    else:
        wam_ddp = wam

    # Color Palette for multi-wm evaluation
    color_palette = create_fixed_color_palette(params.nb_wm_eval)

    # Setup for validation
    validation_augs = [
        (Identity,          [0]),  # No parameters needed for identity
        (HorizontalFlip,    [0]),  # No parameters needed for flip
        (Rotate,            [10, 30, 45, 90]),  # (min_angle, max_angle)
        (Resize,            [0.5, 0.75]),  # size ratio
        (Crop,              [0.5, 0.75]),  # size ratio
        (Perspective,       [0.2, 0.5, 0.8]),  # distortion_scale
        (Brightness,        [0.5, 1.5]),
        (Contrast,          [0.5, 1.5]),
        (Saturation,        [0.5, 1.5]),
        (Hue,               [-0.5, -0.25, 0.25, 0.5]),
        (JPEG,              [40, 60, 80]),
        (GaussianBlur,      [3, 5, 9, 17]),
        (MedianFilter,      [3, 5, 9, 17]),
        (CropResizePad,     [(0.6, 0.8, 0.6, 0.8)]),  # resize_h, resize_w, crop_h, crop_w
    ]
    validation_augs_different_sizes = [
        (Identity,          [0]),
        (UpperLeftCrop,     [0.5]),
    ]
    # Sample validation masks
    dummy_img = torch.ones(3, params.img_size_extractor, params.img_size_extractor)
    validation_masks = augmenter.mask_embedder.sample_representative_masks(dummy_img)  # 5 256, 256
    _, individuals = augmenter.mask_embedder.sample_multiwm_masks(dummy_img, nb_times=params.nb_wm_eval)  # nb_wm_eval, 1, 256, 256
    # Save validation masks if in the main process
    if udist.is_main_process():
        save_image(validation_masks, os.path.join(params.output_dir, 'validation_masks.png'))
        save_image(individuals, os.path.join(params.output_dir, 'validation_masks_multiwm.png'))
    # Evaluation only mode
    if params.only_eval:
        if params.distributed:
            val_loader.sampler.set_epoch(start_epoch)
            val_loader_multi_wm.sampler.set_epoch(start_epoch)
        val_stats = eval_full(wam, val_loader, image_detection_loss, start_epoch, validation_augs, validation_masks, params)
        if params.eval_multiple_w:
            val_stats_kwm = eval_full_kwm(wam, val_loader_multi_wm, image_detection_loss, start_epoch, validation_augs, individuals, params)
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log_only_eval.txt'), 'a') as f:
                f.write(json.dumps(val_stats) + "\n")
            if params.eval_multiple_w:
                with open(os.path.join(params.output_dir, 'log_only_eval_kwm.txt'), 'a') as f:
                    f.write(json.dumps(val_stats_kwm) + "\n")
        return
    # Start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):
        log_stats = {'epoch': epoch}
        # Step the scheduler and scaling scheduler if they exist
        if scheduler is not None:
            scheduler.step(epoch)
        print(f'Epoch {epoch} - scaling_w: {wam.scaling_w}')
        # Set epoch for distributed data loaders
        if params.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            val_loader_multi_wm.sampler.set_epoch(epoch)
        # Train for one epoch
        train_stats = train_one_epoch(wam_ddp, optimizers, train_loader, image_detection_loss, epoch, color_palette, params)
        log_stats = {**log_stats, **{f'train_{k}': v for k, v in train_stats.items()}}
        # Evaluate periodically
        if epoch % params.eval_freq == 0:
            val_stats = eval_full(wam, val_loader, image_detection_loss, epoch, validation_augs, validation_masks, params)
            val_stats_kwm = eval_full_kwm(wam, val_loader_multi_wm, image_detection_loss, epoch, validation_augs, individuals, params) if params.eval_multiple_w else {}
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}, **{f'val_kwm_{k}': v for k, v in val_stats_kwm.items()}}
        # Log statistics if in the main process
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        # Save checkpoints
        save_dict = {
            'epoch': epoch + 1,
            'model': wam.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
        }
        udist.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        if params.saveckpt_freq and epoch % params.saveckpt_freq == 0:
            udist.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth'))
    # Calculate and print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


def train_one_epoch(
    wam: Wam, 
    optimizers: List[torch.optim.Optimizer], 
    train_loader: torch.utils.data.DataLoader, 
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    color_palette: torch.Tensor,
    params: argparse.Namespace,
):
    """
    Train the model for one epoch. This function handles the forward pass, loss computation, 
    backpropagation, and logging of metrics for each batch in the training data loader.
    """

    # Set the model to training mode
    wam.train()

    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")
    h, w = params.img_size_extractor, params.img_size_extractor
    resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

    # Iterate over the training data loader
    for it, (imgs, masks) in enumerate(metric_logger.log_every(train_loader, 10, header)):
        # Move images to the appropriate device
        imgs = imgs.to(device, non_blocking=True)
    
        # Forward pass through the model
        outputs = wam(imgs, masks, no_overlap=params.multiple_w > 0, params=params)
        outputs["preds"] /= params.temperature
        
        # Determine the last layer of the embedder model based on its type
        last_layer = wam.embedder.get_last_layer() if not params.distributed else wam.module.embedder.get_last_layer()

        # Iterate over optimizers for different parts of the model
        for optimizer_idx in [1, 0]:
            # Compute loss and logs
            loss, logs = image_detection_loss(
                imgs, outputs["imgs_w"], 
                outputs["masks"], outputs["msgs"], outputs["preds"], 
                optimizer_idx, epoch, 
                last_layer=last_layer
            )
            # Zero gradients, backpropagate, and update weights
            optimizers[optimizer_idx].zero_grad()
            loss.backward()
            optimizers[optimizer_idx].step()

        # Log bit accuracy if applicable
        if params.nbits > 0:
            bit_accuracy_ = bit_accuracy(
                outputs["preds"][:, 1:, :, :], 
                outputs["msgs"],
                outputs["masks"]
            ).nanmean().item()

        # Extract mask predictions from the outputs
        mask_preds = outputs["preds"][:, 0:1, :, :]  # b 1 h w
        
        # Initialize log statistics with existing logs and additional metrics
        log_stats = {
            **logs,
            'psnr': psnr(outputs["imgs_w"], imgs).mean().item(),
            'lr': optimizers[0].param_groups[0]['lr'],
            'avg_target': outputs["masks"].float().mean().item()
        }
        
        # Compute and log various accuracy and IoU metrics
        for method, mask in zip([""], [mask_preds]):
            log_stats.update({
                f'acc{method}': accuracy(mask, outputs["masks"].max(1).values.float().unsqueeze(1)).mean().item(),
                f'iou_0{method}': iou(mask, outputs["masks"].max(1).values.float().unsqueeze(1), label=0).mean().item(),
                f'iou_1{method}': iou(mask, outputs["masks"].max(1).values.float().unsqueeze(1), label=1).mean().item(),
                f'avg_pred{method}': mask.mean().item(),
                f'norm_avg{method}': torch.norm(mask, p=2).item(),
            })
            # Calculate mean IoU
            log_stats[f'miou{method}'] = (log_stats[f'iou_0{method}'] + log_stats[f'iou_1{method}']) / 2
        
        # Log bit accuracy if applicable
        if params.nbits > 0:
            log_stats['bit_acc'] = bit_accuracy_
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Update metric logger with log statistics
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        # Save images periodically during training  
        colored_masks = masks_to_colored_image(outputs["masks"].float(), color_palette) # b 3 h w

        if epoch % params.saveimg_freq == 0 and it % 200 == 0 and udist.is_main_process():
            # Save original, watermarked, and augmented images, as well as differences and masks
            save_image(unnormalize_img(imgs), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_0_ori.png'), nrow=8)
            save_image(unnormalize_img(outputs["imgs_w"]), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_1_w.png'), nrow=8)
            save_image(create_diff_img(imgs, outputs["imgs_w"]), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_2_diff.png'), nrow=8)
            save_image(unnormalize_img(outputs["imgs_aug"]), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_3_aug.png'), nrow=8)
            # Save predicted and target masks
            save_image(colored_masks.float(), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_4_mask.png'), nrow=8)
            save_image(F.sigmoid(mask_preds / params.temperature), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_5_pred.png'), nrow=8)
    # Synchronize metric logger across processes
    metric_logger.synchronize_between_processes()
    
    # Print averaged statistics for the training epoch
    print("Averaged {} stats:".format('train'), metric_logger)
    
    # Return a dictionary of global average metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_full(
    wam: Wam, 
    val_loader: torch.utils.data.DataLoader, 
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    validation_augs: List,
    validation_masks: List,
    params: argparse.Namespace,
    eval_name="full"
):
    """
    Evaluate the model with one watermark per image for different validation masks and augmentations.
    This function performs watermark embedding, augmentation, and detection, and logs various metrics.
    """

    # Convert validation masks to a list if they are a tensor
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))
    
    # Set the model to evaluation mode
    wam.eval()
    header = 'Val Full - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    # Define the list of masks and augmentations to save
    tosave = [
        f"mask={1}_aug={'crop_0.33'}", 
        f"mask={2}_aug={'resize_0.75'}", 
        f"mask={3}_aug={'upperleftcrop_0.5'}", 
        f"mask={4}_aug={'jpeg_60'}", 
        f"mask={5}_aug={'identity_0'}"
    ] 
    imgs_tosave = []
    h, w = params.img_size_extractor, params.img_size_extractor
    resize = transforms.Resize((h, w))

    aug_metrics = {}
    
    # Iterate over the validation data loader
    for it, (imgs, masks) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        inverse_resize = transforms.Resize((imgs.shape[-2], imgs.shape[-1]), interpolation=transforms.InterpolationMode.BILINEAR)
        inverse_resize_mask = transforms.Resize((masks.shape[-2], masks.shape[-1]), interpolation=transforms.InterpolationMode.NEAREST)
        
        # Combine validation masks and segmentation masks
        if len(masks.shape) != 1:
            validation_masks_and_seg = validation_masks + [masks]
        else:
            validation_masks_and_seg = validation_masks

        # Break if the number of evaluated images exceeds the limit
        if it * params.batch_size_eval >= params.nb_images_eval: break

        # Move images to the appropriate device
        imgs = imgs.to(device, non_blocking=True)
        
        # Generate random messages for watermarking
        msgs = wam.get_random_msg(imgs.shape[0])  # b x k
        msgs = msgs.to(imgs.device)
        
        # Generate watermarked images
        deltas_w = wam.embedder(resize(imgs), msgs)
        imgs_w = wam.scaling_i * imgs + wam.scaling_w * inverse_resize(deltas_w)

        # Apply attenuation if specified
        if wam.attenuation is not None:
            imgs_w = wam.attenuation(imgs, imgs_w)

        # Iterate over each mask in the validation masks and segmentation
        for mask_id, masks in enumerate(validation_masks_and_seg):
            # Move masks to the appropriate device
            masks = masks.to(imgs.device, non_blocking=True)  # 1 h w
            if len(masks.shape) < 4:
                masks = masks.unsqueeze(0).repeat(imgs_w.shape[0], 1, 1, 1)  # b 1 h w
            masks = inverse_resize_mask(masks).float()  # b 1 h w
            
            # Apply watermark masking
            imgs_masked = imgs_w * masks + imgs * (1 - masks)
            
            # Iterate over each transformation and its strengths
            for transform, strengths in validation_augs:
                # Create an instance of the transformation
                transform_instance = transform()

                # Iterate over each strength
                for strength in strengths:
                    h, w = params.img_size_extractor, params.img_size_extractor
                    imgs_aug, masks_aug = transform_instance(imgs_masked, masks, strength)
                    imgs_aug_ori, masks_aug_ori = inverse_resize(imgs_aug), inverse_resize_mask(masks_aug)
                                        
                    # Resize augmented images if necessary
                    if imgs_aug.shape[-2:] != (h, w):
                        imgs_aug = nn.functional.interpolate(imgs_aug, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
                        masks_aug = nn.functional.interpolate(masks_aug, size=(h, w), mode='nearest')
                    
                    # Select the current augmentation
                    selected_aug = str(transform.__name__).lower() + '_' + str(strength)

                    # Detect watermark in augmented images
                    preds = wam.detector(imgs_aug)
                    # Calculate bit accuracy if applicable
                    if params.nbits > 0:
                        bit_preds = preds[:, 1:, :, :]
                        bit_accuracy_ = bit_accuracy(
                            bit_preds, 
                            msgs.unsqueeze(1),
                            masks_aug
                        ).nanmean().item()
                    
                    # Start with masks by using the first bit of the prediction
                    mask_preds = preds[:, 0:1, :, :]  # b 1 h w
                    
                    # Initialize dictionary to store log statistics
                    log_stats = {}
                    
                    # Log bit accuracy if applicable
                    if params.nbits > 0:
                        log_stats[f'bit_acc'] = bit_accuracy_
                    # Compute stats for the augmentation and strength
                    masks_aug = masks_aug.float()
                    for method, mask_preds_ in [('', mask_preds)]:
                        # Log various accuracy and IoU metrics
                        log_stats.update({
                            f'acc{method}': accuracy(mask_preds_, masks_aug).mean().item(),
                            f'iou_0{method}': iou(mask_preds_, masks_aug, label=0).mean().item(),
                            f'iou_1{method}': iou(mask_preds_, masks_aug, label=1).mean().item(),
                            f'avg_pred{method}': torch.sigmoid(mask_preds_).mean().item(),
                            f'avg_pred_hard_{method}': (torch.sigmoid(mask_preds) > 0.5).float().mean().item(),
                            f'avg_target{method}': masks_aug.mean().item(),
                            f'norm_avg{method}': torch.norm(torch.sigmoid(mask_preds_), p=2).item(),
                        })
                        # Calculate mean IoU
                        log_stats[f'miou{method}'] = (log_stats[f'iou_0{method}'] + log_stats[f'iou_1{method}']) / 2
                        
                        # Calculate bit accuracy for different decoding methods if applicable
                        if params.nbits > 0:
                            for decode_method in ['semihard', 'soft']:
                                log_stats[f"bit_acc{method}_{decode_method}"] = bit_accuracy_inference(
                                    bit_preds, 
                                    msgs,
                                    F.sigmoid(mask_preds_),  # b h w
                                    method=decode_method
                                ).nanmean().item()
                    
                    # Create a key for the current mask and augmentation
                    current_key = f"mask={mask_id}_aug={selected_aug}"
                    log_stats = {f"{k}_{current_key}": v for k, v in log_stats.items()}
                    # Save stats of the current augmentation
                    aug_metrics = {**aug_metrics, **log_stats}
                    # Save some of the images if conditions are met
                    masks_aug_ori = masks_aug_ori.float()
                    if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process():
                        if current_key in tosave:
                            idx = len(imgs_tosave) // 5  # consider 1 image per augmentation
                            imgs_tosave.append(unnormalize_img(imgs[idx].cpu()))
                            imgs_tosave.append(unnormalize_img(imgs_w[idx].cpu()))
                            imgs_tosave.append(unnormalize_img(imgs_aug_ori[idx].cpu()))
                            imgs_tosave.append(masks_aug_ori[idx].cpu().repeat(3, 1, 1))
                            imgs_tosave.append(inverse_resize(F.sigmoid(mask_preds[idx]).repeat(3, 1, 1)).cpu())
                            tosave.remove(current_key)
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Update metric logger with augmentation metrics
        for name, loss in aug_metrics.items():
            if name == 'bit_acc' and math.isnan(loss):
                continue
            if name in ["decode_loss", "decode_scale"] and loss == -1:
                continue  # Skip this update or replace with a default value
            metric_logger.update(**{name: loss})
    # Save images if the current epoch is a multiple of the save frequency or if only evaluation is being performed
    if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process():
        aux = "" if not params.only_eval else "_only_eval"
        save_image(torch.stack(imgs_tosave), os.path.join(params.output_dir, f'{epoch:03}_val_{eval_name}{aux}.png'), nrow=5)
    # Synchronize metric logger across processes
    metric_logger.synchronize_between_processes()
    
    # Print averaged statistics for validation
    print("Averaged {} stats:".format('val'), metric_logger)
    
    # Return a dictionary of global average metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def eval_full_kwm(
    wam: Wam, 
    val_loader: torch.utils.data.DataLoader, 
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    validation_augs: List,
    validation_masks: List,
    params: argparse.Namespace,
):
    """
    Evaluate the model with multiple watermarks per image for different validation masks and augmentations.
    This function performs watermark embedding, augmentation, and detection, and logs various metrics.
    """
    # Set the model to evaluation mode
    wam.eval()
    header = 'Val Full Multi WM - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    nb_wm_eval = params.nb_wm_eval
    nb_eval_names = [f"{k}wm" for k in range(1, nb_wm_eval+1)]

    # Initialize dictionaries to store metrics and counters
    aug_metrics = {}  # Define metric_names based on your metrics
    counters = {}
    
    # Dictionary to store images for saving
    imgs_tosave = {"segmentation": {True: [], False: []}, "rectangles": {True: [], False: []}}
    
    # Iterate over the validation data loader
    for it, (imgs, masks) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        # Break if the number of evaluated images exceeds the limit
        if it * params.batch_size_eval >= params.nb_images_eval_multiple_w: break
        
        # Iterate over roll options (True/False)
        for roll in [True, False]:    
            # Iterate over mask types
            for name_mask, masks_ in [("segmentation", masks), ("rectangles", validation_masks)]:   
                if name_mask == "rectangles":
                    # Adjust mask dimensions for rectangles
                    masks_ = masks_.squeeze(1).repeat(imgs.shape[0], 1, 1, 1)
                
                # Move images and masks to the appropriate device
                imgs = imgs.to(device, non_blocking=True)
                masks_ = masks_.to(imgs.device, non_blocking=True) 
                
                msgs_l = []
                first_mask = masks_[:, 0, :, :].unsqueeze(1).float()
                combined_mask = torch.zeros_like(masks_[:, 0, :, :].unsqueeze(1).float())
                combined_imgs = imgs.clone()
                
                # Iterate over the number of watermarks to evaluate
                for nb_wm in range(nb_wm_eval):
                    nb_wm_name = nb_eval_names[nb_wm]
                    mask = masks_[:, nb_wm, :, :].unsqueeze(1).float()
                    combined_mask += mask
                    
                    # Generate random messages for watermarking
                    msgs = wam.get_random_msg(imgs.shape[0])  # b x k
                    msgs = msgs.to(imgs.device)
                    msgs_l.append(msgs)
                    
                    # Embed watermark into images
                    deltas_w = wam.embedder(imgs, msgs)
                    imgs_w = wam.scaling_i * imgs + wam.scaling_w * deltas_w
                    
                    # Apply attenuation if specified
                    if wam.attenuation is not None:
                        imgs_w = wam.attenuation(imgs, imgs_w)
                    
                    # Combine images with watermark based on roll option
                    if not roll:
                        combined_imgs = combined_imgs * (1 - mask) + imgs_w * mask
                    else:
                        combined_imgs = combined_imgs * torch.roll(1 - mask, shifts=-1, dims=0) + torch.roll(imgs_w, shifts=-1, dims=0) * torch.roll(mask, shifts=-1, dims=0)
                    
                    # Detect watermark in combined images
                    preds = wam.detector(combined_imgs)
                    
                    # Start with masks by using the first bit of the prediction
                    mask_preds = preds[:, 0:1, :, :]  # b 1 h w
                    
                    # Initialize dictionary to store log statistics
                    log_stats = {}
                    
                    # Calculate bit accuracy if applicable
                    if params.nbits > 0:
                        bit_preds = preds[:, 1:, :, :]
                        bit_accuracy_ = bit_accuracy(
                            bit_preds, 
                            msgs_l[0].unsqueeze(1) if not roll else torch.roll(msgs_l[0].unsqueeze(1), shifts=-1, dims=0),
                            first_mask if not roll else torch.roll(first_mask, shifts=-1, dims=0)
                        ).nanmean().item()
                        
                        msgs_l_aux = torch.stack(msgs_l)
                        msgs_l_aux = msgs_l_aux.transpose(0, 1)
                        bit_preds = preds[:, 1:, :, :]
                        bit_accuracy_cummulate = bit_accuracy(
                            bit_preds, 
                            msgs_l_aux if not roll else torch.roll(msgs_l_aux, shifts=-1, dims=0),
                            masks_[:, :nb_wm+1, :, :] if not roll else torch.roll(masks_[:, :nb_wm+1, :, :], shifts=-1, dims=0)
                        ).nanmean().item()
                        
                        # Log bit accuracy statistics
                        log_stats[f'bit_acc_{nb_wm_name}_{name_mask}_roll={roll}'] = bit_accuracy_
                        log_stats[f'bit_acc_cummulate_{nb_wm_name}_{name_mask}_roll={roll}'] = bit_accuracy_cummulate

                    # Compute stats for the augmentation and strength
                    for mask_preds_, mask_target in [(mask_preds, combined_mask if not roll else torch.roll(combined_mask, shifts=-1, dims=0))]:
                        mask_target = mask_target.float()
                        # Log various accuracy and IoU metrics
                        log_stats.update({
                            f'acc_{nb_wm_name}_{name_mask}_roll={roll}': accuracy(mask_preds_, mask_target).mean().item(),
                            f'iou_0_{nb_wm_name}_{name_mask}_roll={roll}': iou(mask_preds_, mask_target, label=0).mean().item(),
                            f'iou_1_{nb_wm_name}_{name_mask}_roll={roll}': iou(mask_preds_, mask_target, label=1).mean().item(),
                            f'avg_pred_{nb_wm_name}_{name_mask}_roll={roll}': mask_preds_.mean().item(),
                            f'avg_target_{nb_wm_name}_{name_mask}_roll={roll}': mask_target.mean().item(),
                            f'norm_avg_{nb_wm_name}_{name_mask}_roll={roll}': torch.norm(mask_preds_, p=2).item(),
                        })
                        # Calculate mean IoU
                        log_stats[f'miou_{nb_wm_name}_{name_mask}_roll={roll}'] = (
                            log_stats[f'iou_0_{nb_wm_name}_{name_mask}_roll={roll}'] + 
                            log_stats[f'iou_1_{nb_wm_name}_{name_mask}_roll={roll}']
                        ) / 2
                    
                    # Update augmentation metrics with log statistics
                    for key, value in log_stats.items():
                        if key not in aug_metrics:
                            aug_metrics[key] = 0
                            counters[key] = 0
                        counters[key] += 1
                        aug_metrics[key] += (value - aug_metrics[key]) / counters[key]
                    
                    # Save some of the images if conditions are met
                    if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process() and it == 0:
                        idx = 1
                        imgs_tosave[name_mask][roll].append(unnormalize_img(imgs[idx].cpu()))
                        imgs_tosave[name_mask][roll].append(unnormalize_img(imgs_w[idx if not roll else idx + 1].cpu()))
                        imgs_tosave[name_mask][roll].append(unnormalize_img(combined_imgs[idx].cpu()))
                        imgs_tosave[name_mask][roll].append(mask[idx if not roll else idx + 1].cpu().repeat(3, 1, 1))
                        imgs_tosave[name_mask][roll].append(combined_mask[idx if not roll else idx + 1].cpu().repeat(3, 1, 1))
                        imgs_tosave[name_mask][roll].append(F.sigmoid(mask_preds[idx]).cpu().repeat(3, 1, 1))
                
                # Synchronize CUDA operations
                torch.cuda.synchronize()
                
                # Update metric logger with augmentation metrics
                for name, loss in aug_metrics.items():
                    if name == 'bit_acc' and math.isnan(loss):
                        continue
                    if name in ["decode_loss", "decode_scale"] and loss == -1:
                        continue  # Skip this update or replace with a default value
                    metric_logger.update(**{name: loss})

    # save images
    if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process():
        aux = f"_{nb_wm_eval}_wm" if not params.only_eval else f"_only_eval_{nb_wm_eval}_wm"
        for roll in [False, True]:
            save_image(torch.stack(imgs_tosave["segmentation"][roll]), os.path.join(params.output_dir, f'{epoch:03}_val_full{aux}_segmentation_roll={roll}.png'), nrow=6)
            save_image(torch.stack(imgs_tosave["rectangles"][roll]), os.path.join(params.output_dir, f'{epoch:03}_val_full{aux}_rectangles_roll={roll}.png'), nrow=6)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val multi wm'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
