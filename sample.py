# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
from datetime import timedelta
from PIL import Image

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from models import model_dict
from diffusion import create_diffusion
from vae import AutoencoderKL


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    # Setup DDP:
    if args.distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=7200))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Build and load model:
    assert args.image_size % 16 == 0
    input_size = args.image_size // 16
    vae = AutoencoderKL(ckpt_path=args.tokenizer_path)
    vae = vae.eval().to(device)

    model = model_dict[args.model](
        input_size=input_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt 
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()

    # define diffusion sampler:
    diffusion = create_diffusion(str(args.diffusion_steps))
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Saving .png samples at {args.sample_dir}")
    if args.distributed:
        dist.barrier()

    D = model.hidden_size
    L = model.num_patches
    h = w = int(L ** 0.5)
    c = model.latent_channels
    p = model.patch_size
    cond_len = model.num_cond_tokens
    patch_dim = c * p ** 2
    N = args.per_proc_batch_size
    global_batch_size = N * world_size
    iterations = int(math.ceil(args.num_fid_samples / global_batch_size))
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    for _ in pbar:
        # Sample class inputs:
        y = torch.randint(0, args.num_classes, (N,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            y_null = torch.tensor([1000] * N, device=device)
            y = torch.cat([y, y_null], 0)

        # initialize kv-cache
        kv_cache = {i:{"k": None, "v": None} for i in range(len(model.blocks))}

        # compute class condition and add to kv-cache
        y_embed = model.y_embedder(y, train=False)
        y_embed = y_embed.reshape(-1, cond_len, D)
        cond = y_embed + model.cond_pos_embed
        model.forward_cache_update(cond, kv_cache=kv_cache)

        # generate in pure diffusion mode
        z = torch.randn(N, L, patch_dim, device=device)
        output = diffusion.p_sample_loop(
            model.forward_inference, 
            z, 
            kv_cache, 
            model.pos_embed.repeat(N, 1, 1),
            cfg_scale=args.cfg_scale,
        )
        output = output.reshape(shape=( N, h, w, p, p, c))
        output = torch.einsum("nhwpqc->nchpwq", output)
        output = output.reshape(shape=(N, c, h * p, w * p))

        output = vae.decode(output / 0.2325)
        output = torch.clamp(127.5 * output + 128.0, 0, 255).permute(0, 2, 3, 1)
        output = output.to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(output):
            index = i * world_size + rank + total
            Image.fromarray(sample).save(f"{args.sample_dir}/{index:06d}.png")
        total += global_batch_size

    if args.distributed:
        dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)
        print("Done.")
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(model_dict.keys()))
    parser.add_argument("--tokenizer-path",  type=str)
    parser.add_argument("--sample-dir", type=str, default="results/samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--diffusion-steps", type=str, default="250")
    args = parser.parse_args()
    main(args)
