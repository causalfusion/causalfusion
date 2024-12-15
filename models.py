# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            norm_layer=None,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. 
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class GeneralizedCausalAttention(nn.Module):
    def __init__(self, dim, num_heads, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)

    def _forward_kv_cache(
        self, 
        x: torch.Tensor, 
        layer_index: int,
        kv_cache: dict,
        update_kv_cache: bool = False,
    ):
        N, Lq = x.shape[:2]
        qkv = self.qkv(x).reshape(N, Lq, 3, self.num_heads, self.head_dim)
        q, curr_k, curr_v = qkv.permute(2, 0, 3, 1, 4).unbind(0) # N, nhead, Lq, dhead
        q = self.q_norm(q)
        curr_k = self.k_norm(curr_k)

        if kv_cache[layer_index]["k"] is not None:
            k = kv_cache[layer_index]["k"]
            v = kv_cache[layer_index]["v"]
            k = torch.cat((k, curr_k), dim=2)
            v = torch.cat((v, curr_v), dim=2)
        else:
            k = curr_k
            v = curr_v

        if update_kv_cache:
            kv_cache[layer_index]["k"] = k
            kv_cache[layer_index]["v"] = v

        return self._forward_sdpa(q, k, v, attn_mask=None)

    def _forward(self, x, attn_mask):
        N, L = x.shape[:2]
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0) # N, nhead, L, dhead
        q = self.q_norm(q)
        k = self.k_norm(k)
        return self._forward_sdpa(q, k, v, attn_mask)

    def _forward_sdpa(self, q, k, v, attn_mask):
        N, _, Lq, _ = q.shape
        q = q * self.scale
        k = k.transpose(-2, -1)
        if attn_mask is not None:
            attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(
                attn_mask, float("-inf")
            )
            attn = q @ k + attn_mask
        else:
            attn = q @ k
        attn = attn.softmax(dim=-1) # N, nhead, Lq, L
        x = attn @ v

        x = x.transpose(1, 2).reshape(N, Lq, -1)
        x = self.proj(x)
        return x

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        kv_cache: Optional[dict] = None,
        layer_index: Optional[int] = None,
        update_kv_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        if kv_cache is not None:
            return self._forward_kv_cache(
                x, 
                kv_cache=kv_cache, 
                layer_index=layer_index, 
                update_kv_cache=update_kv_cache
            )
        else:
            return self._forward(x, attn_mask)


class Block(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0, 
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_size)
        self.attn = GeneralizedCausalAttention(
            hidden_size, num_heads=num_heads, norm_layer=norm_layer
        )
        self.norm2 = norm_layer(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = mlp_layer(
            in_features=hidden_size, 
            hidden_features=mlp_hidden_dim, 
            out_features=hidden_size,
        )

    def forward(self, x, **kwargs):
        x = x + self.attn(self.norm1(x), **kwargs)
        x = x + self.mlp(self.norm2(x))
        return x


class CausalDiffusionModel(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        latent_channels=4,
        hidden_size=1152,
        num_cond_tokens=4,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_layer=Mlp,
        norm_layer=nn.LayerNorm,
        class_dropout_prob=0.1,
        num_classes=1000,
        trainable_pos=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_cond_tokens = num_cond_tokens
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.grad_checkpoint = False

        patch_latent_dim = self.patch_size ** 2 * self.latent_channels
        self.x_proj = nn.Linear(patch_latent_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, 
            hidden_size * self.num_cond_tokens, 
            class_dropout_prob
        )
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), 
            requires_grad=trainable_pos
        )
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_cond_tokens, hidden_size)
        )
        self.blocks = nn.ModuleList([
            Block(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio, 
                norm_layer=norm_layer, 
                mlp_layer=mlp_layer
            ) 
            for _ in range(depth)
        ])
        self.final_layer = nn.Sequential(
            norm_layer(hidden_size),
            nn.Linear(hidden_size, patch_latent_dim, bias=True),
        )
        self.initialize_weights()

    def set_grad_checkpoint(self, grad_checkpoint):
        self.grad_checkpoint = grad_checkpoint

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0.0)
                torch.nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        # Initialize label position embedding:
        nn.init.normal_(self.cond_pos_embed, std=0.02)

        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)

    def forward_cache_update(self, x, kv_cache):
        for idx, block in enumerate(self.blocks):
            x = block(x, layer_index=idx, kv_cache=kv_cache, update_kv_cache=True)
        return None

    def _forward_inference(self, xn, t, kv_cache, pos_embed):
        xn = self.x_proj(xn) + pos_embed
        N, _, D = xn.shape
        t_embed = self.t_embedder(t).reshape(N, 1, D)
        xn = xn + t_embed

        for idx, block in enumerate(self.blocks):
            xn = block(xn, layer_index=idx, kv_cache=kv_cache, update_kv_cache=False)

        xn = self.final_layer(xn)
        return xn

    def forward_inference(
        self, xn, t, kv_cache, pos_embed, cfg_scale=1.0, cfg_interval=[0, 1000]
    ):
        assert t.unique().shape[0] == 1
        if cfg_scale > 1.0:
            xn = torch.cat([xn, xn], dim=0)
            t = torch.cat([t, t], dim=0)
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)

        xn = self._forward_inference(xn, t, kv_cache, pos_embed)

        if cfg_scale > 1.0:
            xn_cond, xn_uncond = torch.split(xn, len(xn) // 2, dim=0)
            if cfg_interval[0] <= t[0] < cfg_interval[1]:
                xn = xn_uncond + cfg_scale * (xn_cond - xn_uncond)
            else:
                xn = xn_cond

        return xn

    def forward(self, xn, t, x, y, attn_mask, last_split_size, noise):
        """
        Args:
            xn: noised vae latent
            t: time step
            x: vae latent
            y: condition
            attn_mask: generalized causal attn_mask
        """
        N, c, h, w = x.shape
        p = self.patch_size
        d = p**2 * c

        # patchify and project x and xn
        x = x.reshape(N, c, h // p, p, w // p, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(N, -1, d)
        x = self.x_proj(x)
        x = x + self.pos_embed
        xn = xn.reshape(N, c, h // p, p, w // p, p)
        xn = torch.einsum("nchpwq->nhwpqc", xn)
        xn = xn.reshape(N, -1, d)
        xn = self.x_proj(xn)
        xn = xn + self.pos_embed
        # patchify noise
        noise = noise.reshape(N, c, h // p, p, w // p, p)
        noise = torch.einsum("nchpwq->nhwpqc", noise)
        noise = noise.reshape(N, -1, d)

        # shuffle x and xn
        N, L, D = x.shape
        random_noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(random_noise, dim=1).unsqueeze(-1)
        ids_shuffle_x = ids_shuffle.repeat(1, 1, D)
        x = torch.gather(x, dim=1, index=ids_shuffle_x)
        xn = torch.gather(xn, dim=1, index=ids_shuffle_x)
        x = x[:, :L - last_split_size, :]
        # shuffle noise
        ids_shuffle_n = ids_shuffle.repeat(1, 1, d)
        noise = torch.gather(noise, dim=1, index=ids_shuffle_n)

        # prepare condition tokens
        y_embed = self.y_embedder(y, train=True).reshape(N, self.num_cond_tokens, D)
        cond = y_embed + self.cond_pos_embed
        x = torch.cat((cond, x), dim=1)

        # add time embedding on xn
        t_embed = self.t_embedder(t).reshape(N, 1, D)
        xn = xn + t_embed

        # forward transformer
        x = torch.cat((x, xn), dim=1)
        if self.grad_checkpoint:
            for block in self.blocks:
                x = checkpoint(block, x, attn_mask=attn_mask, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)
        xn = self.final_layer(x[:, -L:, :])

        # return loss
        return (noise - xn) ** 2


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def CausalFusion_L(**kwargs):
    return CausalDiffusionModel(
        depth=24, hidden_size=1024, patch_size=1, latent_channels=16, num_heads=16, 
        num_cond_tokens=64, trainable_pos=True, **kwargs
    )


def CausalFusion_XL(**kwargs):
    return CausalDiffusionModel(
        depth=32, hidden_size=1280, patch_size=1, latent_channels=16, num_heads=20, 
        num_cond_tokens=64, trainable_pos=True, **kwargs
    )


def CausalFusion_H(**kwargs):
    return CausalDiffusionModel(
        depth=48, hidden_size=1408, patch_size=1, latent_channels=16, num_heads=22, 
        num_cond_tokens=64, trainable_pos=True, **kwargs
    )


model_dict = {
    "CausalFusion-L": CausalFusion_L, 
    "CausalFusion-XL": CausalFusion_XL, 
    "CausalFusion-H": CausalFusion_H
}
