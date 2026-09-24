#!/usr/bin/env python3
"""
jepa_model.py

CGM-JEPA encoder (Muhammad et al., arXiv:2605.00933), re-implemented here so the
published checkpoints load without taking a dependency on the authors' repo.

The upstream model is a representation learner, not a forecaster: its pretraining
objective is an L1 loss between predicted and EMA-target patch embeddings, and only
the encoder is published. There is no decoder from latent space back to mg/dL, so a
forecast requires a readout head fitted on top of frozen embeddings (see
predict_jepa.py).

Architecture, fixed by the published config.json:
    input            (B, 24, 12)  one day of 5-min CGM as 24 hourly patches, raw mg/dL
    ValueEmbedding   Conv1d(1, 96, k=3, s=3) per patch -> flatten 384 -> Linear -> 96
    + sinusoidal positional embedding
    3 x pre-norm Transformer block, 96-d, 6 heads, MLP ratio 4
    LayerNorm
    output           (B, 24, 96) per-patch embeddings

Time-feature embedding is present in the checkpoint but was disabled during
pretraining (use_time_feature=False), so it is never applied here either: feeding
it would push inputs off the distribution the weights were fitted on.

The `proj` head (96 -> 1024 -> 48) is carried so state dicts load strictly, but it
is not on either pretraining loss path and its outputs are not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

PATCH_LEN = 12          # 12 x 5 min = 1 hour
NUM_PATCHES = 24        # 24 hours
HIST_LEN = PATCH_LEN * NUM_PATCHES

ENCODERS = ("x_cgm_jepa", "cgm_jepa")
HF_REPO = "CRUISEResearchGroup/CGM-JEPA"


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, qk_scale)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

    def forward(self, x_len):
        return self.pos_emb[:, :x_len]


class ValueEmbedding(nn.Module):
    """Per-patch strided conv, flattened and projected to the model dimension."""

    def __init__(self, dim, in_channels, patch_size, bias=True):
        super().__init__()
        self.proj = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        conv_len = (in_channels - patch_size) // patch_size + 1
        self.fc = nn.Linear(dim * conv_len, dim)

    def forward(self, x):
        B, N, L = x.shape
        x = self.proj(x.reshape(-1, 1, L)).reshape(B * N, -1)
        return self.fc(x).view(B, N, -1)


class TimeFeatureEmbedding(nn.Module):
    """Unused at inference; kept so published state dicts load strictly."""

    def __init__(self, d_model, d_inp):
        super().__init__()
        self.proj = nn.Linear(d_inp, d_model)


class DataEmbedding(nn.Module):
    def __init__(self, dim, in_channels, patch_size, time_inp_dim):
        super().__init__()
        self.value_embedding = ValueEmbedding(dim, in_channels, patch_size)
        self.positional_embedding = PositionalEmbedding(dim)
        self.timefeature_embedding = TimeFeatureEmbedding(dim, time_inp_dim)

    def forward(self, x):
        val = self.value_embedding(x)
        return val + self.positional_embedding(val.size(1))


class CGMJepaEncoder(nn.Module):
    def __init__(
        self,
        dim_in=PATCH_LEN,
        kernel_size=3,
        embed_dim=96,
        nhead=6,
        num_layers=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        time_inp_dim=5,
        **_ignored,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.data_embedding = DataEmbedding(embed_dim, dim_in, kernel_size, time_inp_dim)
        self.predictor_blocks = nn.ModuleList(
            [Block(embed_dim, nhead, mlp_ratio, qkv_bias, qk_scale) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.proj = MLP(embed_dim, 1024, 48)

    def forward(self, x):
        """x: (B, 24, 12) raw CGM patches -> (B, 24, embed_dim) patch embeddings."""
        x = self.data_embedding(x)
        for blk in self.predictor_blocks:
            x = blk(x)
        return self.encoder_norm(x)


def resolve_device(spec="auto"):
    """
    Pick a torch device, and pin float32 matmul precision when it is a GPU.

    TF32 turns float32 matmuls into 10-bit-mantissa operations on Ampere and
    later. Left enabled, the same window embeds differently on a GPU node than
    on a CPU node, so a results file would silently depend on where the job
    landed. The speed it buys is irrelevant for a 522 k-parameter encoder.
    """
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError:
            pass
    return device


@torch.no_grad()
def check_device_parity(model, device, n=256, atol=2e-4):
    """
    Assert the accelerator reproduces the CPU forward pass before a run uses it.

    A silent numerical divergence between nodes is the failure this guards: it
    does not raise, it just makes two results files incomparable. Cheap enough
    to run at startup every time.
    """
    if torch.device(device).type == "cpu":
        return 0.0
    g = torch.Generator().manual_seed(0)
    x = (120 + 60 * torch.rand((n, NUM_PATCHES, PATCH_LEN), generator=g)).float()
    on_cpu = model.to("cpu")(x)
    on_dev = model.to(device)(x.to(device)).cpu()
    diff = float((on_cpu - on_dev).abs().max())
    if diff > atol:
        raise RuntimeError(
            f"{device} embeddings diverge from CPU by {diff:.2e} (tolerance {atol:.0e}). "
            f"Results would depend on which node the job landed on. Re-run with --device cpu."
        )
    return diff


def load_encoder(weights_dir, name="x_cgm_jepa", device="cpu"):
    """Build the encoder from the published config.json and load its safetensors."""
    from safetensors.torch import load_file

    d = Path(weights_dir) / name
    if not d.is_dir():
        raise FileNotFoundError(
            f"No weight directory at {d}. Fetch the published checkpoints with:\n"
            f"  huggingface-cli download {HF_REPO} --local-dir {weights_dir} "
            f"--include 'cgm_jepa/*' 'x_cgm_jepa/*'"
        )
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model = CGMJepaEncoder(**cfg)
    missing, unexpected = model.load_state_dict(load_file(d / "model.safetensors"), strict=False)
    # pos_emb is a deterministic sinusoidal buffer; the checkpoint copy is identical.
    hard_missing = [k for k in missing if "pos_emb" not in k]
    if hard_missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={hard_missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg


@torch.no_grad()
def embed_windows(model, hist, batch_size=512, device="cpu"):
    """
    hist: float32 array (N, 288) of raw mg/dL, most recent sample last, no NaNs.
    Returns (N, 24, embed_dim) float32 patch embeddings.
    """
    out = []
    for i in range(0, len(hist), batch_size):
        chunk = torch.from_numpy(hist[i : i + batch_size]).float()
        chunk = chunk.view(-1, NUM_PATCHES, PATCH_LEN).to(device)
        out.append(model(chunk).cpu().numpy())
    import numpy as np

    return np.concatenate(out, axis=0) if out else np.zeros((0, NUM_PATCHES, model.embed_dim), "float32")
