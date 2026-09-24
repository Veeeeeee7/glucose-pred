#!/usr/bin/env python3
"""
covariates.py

Insulin and carbohydrate features at a forecast origin, shared by every driver
that feeds covariates to a head: probe_covariates.py (frozen-encoder diagnostic)
and finetune_jepa.py (end-to-end fine-tuning).

`Covariates` is an extra-feature hook for jepa_windows.build_fit_set /
iter_eval_chunks: it reads bolus, basal and carbs (never CGM) from a
participant frame and returns one row per origin. `cgm_features` summarises a
filled 24 h CGM history, all at or before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jepa_windows import HORIZONS, STEP_NS, TIME

CHANNELS = ("bolus", "basal", "carbs")
HIST_BINS = ((0, 30), (30, 60), (60, 120), (120, 240), (240, 360))
FUT_BINS = ((0, 30), (30, 60), (60, 90), (90, 120))
AVAIL_MIN = 360
RAW_CGM = 24                     # last 2 h of the filled history

# Action features. Doses act on glucose with a lag, so besides raw 5-min slots
# the hook gives each dose's expected action inside (t, t+h] under a
# first-order exponential response with time constant tau: a past dose at t-s
# contributes exp(-s/tau) - exp(-(s+h)/tau), a future dose at t+s contributes
# 1 - exp(-(h-s)/tau). Two taus per channel let the head interpolate the shape.
TAUS = {"bolus": (55.0, 120.0), "basal": (55.0, 120.0), "carbs": (30.0, 90.0)}
HIST_RAW = 12                    # last hour, 5-min slots
FUT_RAW = 24                     # next 2 h, 5-min slots
EFFECT_PAST = 72                 # 6 h of past doses feed the action features


class Covariates:
    """
    Extra-feature hook for jepa_windows: covariate features at each origin.

    Column blocks, in order (`slices` names them):
      hist_bins  CHANNELS x HIST_BINS window sums
      avail      CHANNELS observed fraction over the last AVAIL_MIN
      fut_bins   CHANNELS x FUT_BINS window sums
      hist_raw   CHANNELS x the last HIST_RAW 5-min slots (t first)
      fut_raw    CHANNELS x the next FUT_RAW 5-min slots (t+5 first)
      effect     CHANNELS x taus x (past, future) x HORIZONS expected action
      tod        sin, cos of time of day at t
    The first three blocks are what hist_cols / fut_cols(h) select; the rest
    only enter through rich_cols(h).
    """

    columns = CHANNELS

    def __init__(self):
        C = len(CHANNELS)
        layout = [("hist_bins", C * len(HIST_BINS)), ("avail", C), ("fut_bins", C * len(FUT_BINS)),
                  ("hist_raw", C * HIST_RAW), ("fut_raw", C * FUT_RAW),
                  ("effect", C * 2 * 2 * len(HORIZONS)), ("tod", 2)]
        self.slices, o = {}, 0
        for name, k in layout:
            self.slices[name] = np.arange(o, o + k)
            o += k
        self.n = o
        self.hist_cols = np.concatenate([self.slices["hist_bins"], self.slices["avail"]])
        self._fut0 = int(self.slices["fut_bins"][0])

    def _effect_col(self, c, k_tau, kind, j):
        return int(self.slices["effect"][0]) + ((c * 2 + k_tau) * 2 + kind) * len(HORIZONS) + j

    def rich_cols(self, horizon):
        """Everything a horizon-h head may use: no raw slot or bin past t+h."""
        j = HORIZONS.index(horizon)
        fut_raw = [int(self.slices["fut_raw"][0]) + c * FUT_RAW + m
                   for c in range(len(CHANNELS)) for m in range(FUT_RAW) if 5 * (m + 1) <= horizon]
        effect = [self._effect_col(c, k, kind, j)
                  for c in range(len(CHANNELS)) for k in range(2) for kind in range(2)]
        return np.concatenate([self.hist_cols, self.fut_cols(horizon), self.slices["hist_raw"],
                               np.array(fut_raw, np.int64), np.array(effect, np.int64),
                               self.slices["tod"]])

    def fut_cols(self, horizon):
        """Future columns whose bins end by t + horizon."""
        keep = [b for b, (_, hi) in enumerate(FUT_BINS) if hi <= horizon]
        return np.array([self._fut0 + c * len(FUT_BINS) + b
                         for c in range(len(CHANNELS)) for b in keep], dtype=np.int64)

    def __call__(self, frame, origins_ns):
        f = frame.sort_values(TIME).drop_duplicates(subset=[TIME], keep="last")
        ts = pd.to_datetime(f[TIME]).to_numpy().astype("datetime64[ns]").astype("int64")
        t0 = ts[0]
        slot = (ts - t0) // STEP_NS
        on_grid = (ts - t0) % STEP_NS == 0
        n_slots = int(slot[-1]) + 1
        i = ((origins_ns - t0) // STEP_NS).astype(np.int64)

        def window_sum(cum, lo, hi):          # sum over slots [lo, hi)
            return cum[np.clip(hi, 0, n_slots)] - cum[np.clip(lo, 0, n_slots)]

        out = np.zeros((len(origins_ns), self.n), np.float64)
        avail_col = len(CHANNELS) * len(HIST_BINS)
        for c, ch in enumerate(CHANNELS):
            v = pd.to_numeric(f[ch], errors="coerce").to_numpy(np.float64)
            grid = np.zeros(n_slots)
            seen = np.zeros(n_slots)
            ok = on_grid & np.isfinite(v)
            grid[slot[ok]] = v[ok]
            seen[slot[ok]] = 1.0
            cum = np.concatenate([[0.0], np.cumsum(grid)])
            cum_seen = np.concatenate([[0.0], np.cumsum(seen)])
            step = 5
            for b, (a, z) in enumerate(HIST_BINS):   # (t-z, t-a]
                out[:, c * len(HIST_BINS) + b] = window_sum(cum, i - z // step + 1, i - a // step + 1)
            w = AVAIL_MIN // step
            out[:, avail_col + c] = window_sum(cum_seen, i - w + 1, i + 1) / w
            for b, (a, z) in enumerate(FUT_BINS):    # (t+a, t+z]
                out[:, self._fut0 + c * len(FUT_BINS) + b] = window_sum(cum, i + a // step + 1, i + z // step + 1)

            # Slot-level views: past slots t, t-5, ... and future t+5, t+10, ...
            def slots(idx):                  # grid values, zero outside the record
                return np.where((idx >= 0) & (idx < n_slots), grid[np.clip(idx, 0, n_slots - 1)], 0.0)
            past = slots(i[:, None] - np.arange(EFFECT_PAST)[None, :])
            fut = slots(i[:, None] + np.arange(1, FUT_RAW + 1)[None, :])
            out[:, self.slices["hist_raw"][c * HIST_RAW:(c + 1) * HIST_RAW]] = past[:, :HIST_RAW]
            out[:, self.slices["fut_raw"][c * FUT_RAW:(c + 1) * FUT_RAW]] = fut
            s_past = step * np.arange(EFFECT_PAST, dtype=np.float64)
            s_fut = step * np.arange(1, FUT_RAW + 1, dtype=np.float64)
            for k, tau in enumerate(TAUS[ch]):
                for j, h in enumerate(HORIZONS):
                    w_past = np.exp(-s_past / tau) - np.exp(-(s_past + h) / tau)
                    w_fut = np.where(s_fut <= h, 1.0 - np.exp(-(h - s_fut) / tau), 0.0)
                    out[:, self._effect_col(c, k, 0, j)] = past @ w_past
                    out[:, self._effect_col(c, k, 1, j)] = fut @ w_fut

        day = ((origins_ns // 1_000_000_000) % 86400) / 86400.0 * 2 * np.pi
        out[:, self.slices["tod"]] = np.stack([np.sin(day), np.cos(day)], axis=1)
        return out


def cgm_features(hist):
    """
    Summary features of the filled 24 h history for the rich variants: the last
    2 h raw, the change over the last 5/15/30/60 min, the spread of the last hour,
    and the day's mean. All at or before t.
    """
    g = hist.astype(np.float64)
    now = g[:, -1:]
    deltas = now - g[:, [-2, -4, -7, -13]]
    return np.hstack([g[:, -RAW_CGM:], deltas, g[:, -12:].std(axis=1, keepdims=True),
                      g.mean(axis=1, keepdims=True)])
