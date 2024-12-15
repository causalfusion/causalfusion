# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Causal Diffusion.
"""
import argparse
import logging
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from time import time

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import get_constant_schedule_with_warmup

from models import model_dict
from diffusion import create_diffusion
from vae import AutoencoderKL


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    TODO: Consider applying only to params that require_grad to avoid small
          numerical changes of pos_embed
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    TODO: need a more elegant one
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def split_integer_exp_decay(S, ar_step_decay=1.0):
    if ar_step_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ar_step_decay) / (1 - math.pow(ar_step_decay, S))
        p = [base * math.pow(ar_step_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]

    # random sample the cumsum to create random AR step sizes
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def get_ar_weights(split_sizes, cumsum, max_weight, min_weight, schedule="linear"):
    assert max_weight >= min_weight
    if max_weight == min_weight:
        return torch.tensor(min_weight)

    weights = []
    full_len = cumsum[-1]
    if schedule == "cosine":
        for size, x in zip(split_sizes, cumsum[:-1]):
            weights.extend(
                size * [
                    math.cos(math.pi / full_len * x) * (max_weight - min_weight) / 2 + 
                    (max_weight + min_weight) / 2
                ]
            )
    elif schedule == "linear":
        for size, x in zip(split_sizes, cumsum[:-1]):
            weights.extend(size * [max_weight - (max_weight - min_weight) / full_len * x])
    else:
        raise NotImplementedError

    return torch.tensor(weights)


def get_attn_mask(sample_len, cond_len, split_sizes=None, cumsum=None):
    visiable_len = sample_len - split_sizes[-1]
    ctx_len = cond_len + visiable_len
    seq_len = ctx_len + sample_len

    attn_mask = torch.ones(size=(seq_len, seq_len))
    attn_mask[:, :cond_len] = 0

    # build `triangle` masks
    triangle1 = torch.ones(size=(visiable_len, visiable_len))
    triangle2 = torch.ones(size=(sample_len, visiable_len))
    triangle3 = torch.ones(size=(sample_len, sample_len))
    for i in range(len(split_sizes) - 1):
        triangle1[cumsum[i]:cumsum[i+1], 0:cumsum[i+1]] = 0
        triangle2[cumsum[i+1]:cumsum[i+2], 0:cumsum[i+1]] = 0
    for i in range(len(split_sizes)):
        triangle3[cumsum[i]:cumsum[i+1], cumsum[i]:cumsum[i+1]] = 0

    # copy mask to attention mask
    attn_mask[cond_len:ctx_len, cond_len:ctx_len] = triangle1
    attn_mask[ctx_len:, cond_len:ctx_len] = triangle2
    attn_mask[ctx_len:, ctx_len:] = triangle3

    return attn_mask[None, None, :, :]


def main(args):
    assert torch.cuda.is_available()
    # Setup DDP:
    if args.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    assert args.global_batch_size % world_size == 0
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        checkpoint_dir = f"{args.results_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.results_dir, rank)
        logger.info(f"Experiment directory created at {args.results_dir}")
    else:
        logger = create_logger(None, rank)

    assert args.image_size % 16 == 0
    input_size = args.image_size // 16
    vae = AutoencoderKL(ckpt_path=args.tokenizer_path).to(device)

    model = model_dict[args.model](
        input_size=input_size,
        num_classes=args.num_classes,
    )
    model.set_grad_checkpoint(args.grad_checkpoint)
    ema = deepcopy(model)
    requires_grad(ema, False)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
    ema = ema.to(device)
    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[device])
    model_without_ddp = model.module if args.distributed else model

    diffusion = create_diffusion(timestep_respacing="")

    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        eps=args.eps, 
        weight_decay=0
    )
    if args.ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    scheduler = get_constant_schedule_with_warmup(
        optimizer=opt, num_warmup_steps=args.warmup_steps
    )
    if args.ckpt is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # Setup data:
    train_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = ImageFolder(args.train_data_path, transform=train_transform)

    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        shuffle = None
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // world_size),
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Prepare models for training:
    update_ema(ema, model_without_ddp, decay=0)
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    train_steps = ckpt["train_steps"] if args.ckpt is not None else 0
    start_epoch = ckpt["start_epoch"] if args.ckpt is not None else 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    patch_size = model_without_ddp.patch_size
    cond_len = model_without_ddp.num_cond_tokens

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        logger.info(f"Training epoch {epoch}...")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).sample().mul_(0.2325)

            h, w = x.shape[2:]
            L = h // patch_size * w // patch_size

            split_sizes, cumsum = split_integer_exp_decay(L, args.ar_step_decay)
            attn_mask = get_attn_mask(L, cond_len, split_sizes, cumsum)
            attn_mask = attn_mask.bool().to(x.device)
            ar_weights = get_ar_weights(
                split_sizes, 
                cumsum, 
                args.max_ar_weight, 
                args.min_ar_weight, 
                args.ar_weight_schedule
            )
            ar_weights = ar_weights.to(x.device).reshape(1, -1, 1)

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(
                    model, x, t, y, 
                    attn_mask=attn_mask, 
                    last_split_size=split_sizes[-1]
                )
                loss = (loss_dict["loss"] * ar_weights).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            update_ema(ema, model_without_ddp)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if args.distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"lr: {opt.param_groups[0]['lr']:.6f}"
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_steps": train_steps,
                        "start_epoch": epoch + 1,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if args.distributed:
                    dist.barrier()

    logger.info("Done!")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-15)
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(model_dict.keys()))
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=25000)
    parser.add_argument("--global-batch-size", type=int, default=2048)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=12500)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-ar-weight", type=float, default=2.0)
    parser.add_argument("--min-ar-weight", type=float, default=1.0)
    parser.add_argument("--ar-weight-schedule", type=str, default="linear")
    parser.add_argument("--ar-step-decay", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
