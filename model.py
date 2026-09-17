#!/usr/bin/env python3
"""
model.py

Causal Gaussian dual-stream glucose forecaster.

History:
    24 h (288 x 5 min) of CGM + insulin + basal + bolus + carbs

1) Mask-aware instance normalization of CGM.
2) Learnable one-sided Gaussian filter:
       state_t = causal low-frequency trend
       event_t = normalized_CGM_t - state_t
3) One-hour state/event patches + multimodal history covariates.
4) Transformer history encoder.
5) Autoregressive latent rollout over 24 future five-minute steps:
       (S_k,E_k,U_{k+1}) -> (S_{k+1},E_{k+1}) -> G_{k+1}

The future transition is causal: prediction k receives only the future covariate
for that step and the latent state propagated from earlier steps.
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class CausalGaussianDecomposer(nn.Module):
    def __init__(self, sigma_min=2.0, sigma_max=12.0, sigma_init=6.0):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # GlucoFM truncates at 3*sigma_max = 36 grid steps.
        self.max_lag = int(math.ceil(3 * sigma_max))

        p = (sigma_init - sigma_min) / (sigma_max - sigma_min)
        p = min(max(p, 1e-5), 1 - 1e-5)
        rho = math.log(p / (1 - p))
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))

    @property
    def sigma(self):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.rho)

    def forward(self, x, mask):
        """
        x, mask: [B,T]. x should already be normalized and missing positions filled with 0.
        """
        r = torch.arange(self.max_lag + 1, device=x.device, dtype=x.dtype)
        sigma = self.sigma.to(dtype=x.dtype)
        w = torch.exp(-(r * r) / (2 * sigma * sigma))
        w = w / w.sum()

        # conv1d performs cross-correlation. Flip so kernel implements
        # y[t] = sum_r w[r] * x[t-r].
        kernel = w.flip(0).view(1, 1, -1)

        xm = x * mask
        num = F.conv1d(
            F.pad(xm.unsqueeze(1), (self.max_lag, 0)),
            kernel,
        ).squeeze(1)
        den = F.conv1d(
            F.pad(mask.unsqueeze(1), (self.max_lag, 0)),
            kernel,
        ).squeeze(1)

        state = num / den.clamp_min(1e-6)
        event = (x - state) * mask
        return state, event


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class DualStreamForecaster(nn.Module):
    def __init__(
        self,
        d_stream=64,
        d_model=128,
        n_heads=4,
        n_layers=3,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.decomposer = CausalGaussianDecomposer()

        # State patch: 12 waveform + 12 observation-mask values + mean + std + end-start difference.
        self.state_embed = MLP(27, 128, d_stream, dropout)

        # Event patch: 12 residual waveform + 12 causal first differences
        # + 12 observation-mask values + mask-aware mean + std.
        self.event_embed = MLP(38, 128, d_stream, dropout)

        # History covariates: 12x4 transformed values + 12x4 masks.
        self.hist_cov_embed = MLP(96, 128, d_stream, dropout)

        self.fuse = nn.Linear(d_stream * 3, d_model)
        self.time_embed = nn.Linear(2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Build initial latent state/event from both last and mean-pooled history context.
        self.s0_head = MLP(2 * d_model, d_model, d_stream, dropout)
        self.e0_head = MLP(2 * d_model, d_model, d_stream, dropout)

        # Future input: 4 values + 4 availability masks + sin/cos time-of-day.
        fut_dim = 10
        trans_in = d_stream * 2 + fut_dim

        self.state_transition = MLP(trans_in, 128, d_stream, dropout)
        self.event_transition = MLP(trans_in, 128, d_stream, dropout)
        self.state_norm = nn.LayerNorm(d_stream)
        self.event_norm = nn.LayerNorm(d_stream)

        self.glucose_head = MLP(d_stream * 2, 128, 1, dropout)

    @staticmethod
    def masked_normalize_cgm(cgm, mask):
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (cgm * mask).sum(dim=1, keepdim=True) / count
        var = (((cgm - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var + 1e-5).clamp_min(5.0)
        z = ((cgm - mean) / std) * mask
        return z, mean, std

    @staticmethod
    def circular_time(minutes):
        angle = 2 * math.pi * (minutes / 1440.0)
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    @staticmethod
    def compress_covariates(x):
        # Robust scale compression without requiring global dataset statistics.
        # Preserves sign in case a source contains signed derived values.
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, hist, fut_cov, origin_minute):
        """
        hist:          [B,288,5]  CGM, insulin, basal, bolus, carbs; NaNs allowed
        fut_cov:       [B,24,4]   insulin, basal, bolus, carbs; NaNs allowed
        origin_minute: [B]        minute-of-day at t

        returns:
            pred [B,24] in mg/dL
        """
        hist_mask = torch.isfinite(hist)
        hist_x = torch.nan_to_num(hist, nan=0.0)

        cgm = hist_x[:, :, 0]
        cgm_mask = hist_mask[:, :, 0].float()

        cgm_z, cgm_mean, cgm_std = self.masked_normalize_cgm(cgm, cgm_mask)
        state, event = self.decomposer(cgm_z, cgm_mask)

        B = hist.shape[0]
        # [B,24,12]
        state_p = state.view(B, 24, 12)
        event_p = event.view(B, 24, 12)
        cgm_mask_p = cgm_mask.view(B, 24, 12)

        obs_count = cgm_mask_p.sum(dim=-1, keepdim=True).clamp_min(1.0)

        state_mean = (state_p * cgm_mask_p).sum(dim=-1, keepdim=True) / obs_count
        state_var = (((state_p - state_mean) * cgm_mask_p) ** 2).sum(
            dim=-1, keepdim=True
        ) / obs_count
        state_std = torch.sqrt(state_var + 1e-6)
        state_diff = (state_p[:, :, -1] - state_p[:, :, 0]).unsqueeze(-1)
        state_feat = torch.cat(
            [state_p, cgm_mask_p, state_mean, state_std, state_diff], dim=-1
        )

        # Causal within-patch first difference; first position uses zero.
        event_diff = torch.zeros_like(event_p)
        event_diff[:, :, 1:] = event_p[:, :, 1:] - event_p[:, :, :-1]
        event_mean = (event_p * cgm_mask_p).sum(dim=-1, keepdim=True) / obs_count
        event_var = (((event_p - event_mean) * cgm_mask_p) ** 2).sum(
            dim=-1, keepdim=True
        ) / obs_count
        event_std = torch.sqrt(event_var + 1e-6)
        event_feat = torch.cat(
            [event_p, event_diff, cgm_mask_p, event_mean, event_std], dim=-1
        )

        # History non-CGM covariates.
        hcov = self.compress_covariates(hist_x[:, :, 1:]).view(B, 24, 12, 4)
        hcov_mask = hist_mask[:, :, 1:].float().view(B, 24, 12, 4)
        cov_feat = torch.cat(
            [hcov.reshape(B, 24, 48), hcov_mask.reshape(B, 24, 48)],
            dim=-1,
        )

        s_tok = self.state_embed(state_feat)
        e_tok = self.event_embed(event_feat)
        c_tok = self.hist_cov_embed(cov_feat)
        tokens = self.fuse(torch.cat([s_tok, e_tok, c_tok], dim=-1))

        # Patch midpoints. First historical sample is t-1435 min.
        patch_idx = torch.arange(24, device=hist.device, dtype=hist.dtype)
        hist_minutes = (
            origin_minute.to(hist.dtype).unsqueeze(1)
            - 1435.0
            + patch_idx.unsqueeze(0) * 60.0
            + 27.5
        ) % 1440.0
        tokens = tokens + self.time_embed(self.circular_time(hist_minutes))

        z = self.encoder_norm(self.encoder(tokens))
        summary = torch.cat([z[:, -1], z.mean(dim=1)], dim=-1)
        s = self.s0_head(summary)
        e = self.e0_head(summary)

        fut_mask = torch.isfinite(fut_cov)
        fut_x = self.compress_covariates(torch.nan_to_num(fut_cov, nan=0.0))

        preds_z = []
        for k in range(24):
            minute_k = (
                origin_minute.to(hist.dtype) + float(5 * (k + 1))
            ) % 1440.0
            time_k = self.circular_time(minute_k)

            u = torch.cat(
                [fut_x[:, k], fut_mask[:, k].float(), time_k],
                dim=-1,
            )
            joint = torch.cat([s, e, u], dim=-1)

            # Simultaneous residual update from the previous state.
            ds = self.state_transition(joint)
            de = self.event_transition(joint)
            s = self.state_norm(s + ds)
            e = self.event_norm(e + de)

            g_z = self.glucose_head(torch.cat([s, e], dim=-1)).squeeze(-1)
            preds_z.append(g_z)

        pred_z = torch.stack(preds_z, dim=1)
        pred = pred_z * cgm_std + cgm_mean
        return pred

