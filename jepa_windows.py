#!/usr/bin/env python3
"""
jepa_windows.py

Shared plumbing for the CGM-JEPA forecasting baseline: turn MetaboNet parquet
into (24 h history, future CGM) pairs on a strict 5-minute grid.

Both the fit set (train.parquet) and the eval set (test.parquet, restricted to
the rows a competition template asks for) go through the same extraction, so the
only difference between them is which parquet and which origins.

Row groups in both parquets carry (source_file, id) statistics, and rows are
stored grouped by participant. That lets a caller read a handful of participants
without scanning the whole file, which is what makes a laptop-sized subsample
run in seconds instead of minutes.

Missingness: an origin is kept only if the sample at t exists and the 24 h
window is at least `min_valid_frac` observed. Interior gaps are linearly
interpolated by default rather than marked with a sentinel, because the encoder
was pretrained on raw mg/dL and a -1 inside a 100-300 mg/dL series is far
outside the value range its input convolution ever saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["source_file", "id"]
TIME = "date"
CGM = "CGM"

HIST_LEN = 288                       # 288 x 5 min = 24 h
STEP_NS = 5 * 60 * 1_000_000_000
HORIZONS = (30, 60, 90, 120)
HORIZON_STEPS = tuple(h // 5 for h in HORIZONS)
GATHER_BLOCK = 8192                  # origins per grid lookup; bounds its temporaries


# --------------------------------------------------------------------------
# Row-group selection
# --------------------------------------------------------------------------

def row_group_bounds(path):
    """[(row_group_index, source_file_min, source_file_max, id_min, id_max)]."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    i_sf, i_id = names.index("source_file"), names.index("id")
    out = []
    for rg in range(pf.metadata.num_row_groups):
        r = pf.metadata.row_group(rg)
        s_sf, s_id = r.column(i_sf).statistics, r.column(i_id).statistics
        if s_sf is None or s_id is None:
            out.append((rg, None, None, None, None))
        else:
            out.append((rg, s_sf.min, s_sf.max, s_id.min, s_id.max))
    return out


def row_groups_for(path, wanted):
    """Row groups whose (source_file, id) statistics can overlap `wanted` keys."""
    keep = []
    for rg, sf_lo, sf_hi, id_lo, id_hi in row_group_bounds(path):
        if sf_lo is None:
            keep.append(rg)
            continue
        for sf, pid in wanted:
            if sf_lo <= sf <= sf_hi and id_lo <= pid <= id_hi:
                keep.append(rg)
                break
    return keep


def spread_row_groups(path, n):
    """`n` row groups spaced evenly through the file, for cross-dataset coverage."""
    total = pq.ParquetFile(path).metadata.num_row_groups
    n = max(1, min(n, total))
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, n)})


def iter_participant_frames(path, row_groups, columns=(TIME, CGM), wanted=None):
    """
    Yield ((source_file, id), frame) for each contiguous participant block in the
    selected row groups. A block split across two row groups is carried forward and
    emitted once whole, so a window is never built across a discontinuity.
    """
    pf = pq.ParquetFile(path)
    cols = list(dict.fromkeys(KEY + list(columns)))
    carry = None
    prev_rg = None

    def emit(block):
        key = (str(block["source_file"].iloc[0]), str(block["id"].iloc[0]))
        if wanted is None or key in wanted:
            return key, block
        return None

    for rg in row_groups:
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        if df.empty:
            continue
        df["source_file"] = df["source_file"].astype(str)
        df["id"] = df["id"].astype(str)
        if carry is not None:
            if prev_rg is not None and rg == prev_rg + 1:
                df = pd.concat([carry, df], ignore_index=True)
            else:
                got = emit(carry)      # discontiguous: the held tail stands alone
                if got:
                    yield got
            carry = None
        prev_rg = rg

        keys = list(zip(df["source_file"].to_numpy(), df["id"].to_numpy()))
        cuts = [0] + [j for j in range(1, len(keys)) if keys[j] != keys[j - 1]] + [len(df)]
        for b in range(len(cuts) - 2):
            got = emit(df.iloc[cuts[b] : cuts[b + 1]])
            if got:
                yield got
        carry = df.iloc[cuts[-2] :].copy()

    if carry is not None and not carry.empty:
        got = emit(carry)
        if got:
            yield got


# --------------------------------------------------------------------------
# Window extraction
# --------------------------------------------------------------------------

def to_ns(values):
    """
    Timestamps as int64 nanoseconds.

    The datetime unit pandas hands back depends on its version and on how the column
    was built: a 5-minute range can arrive as datetime64[us] rather than [ns]. An
    unqualified astype("int64") would then return microseconds, and every exact-grid
    lookup in _gather would miss, silently yielding zero windows. Pin the unit here.
    """
    return pd.to_datetime(values).to_numpy().astype("datetime64[ns]").astype("int64")


def grid_of(frame):
    """
    Sorted unique timestamps (int64 ns) and matching CGM values.

    Built once per participant and reused across chunks: it sorts and
    de-duplicates the whole frame, so calling it per chunk would make a chunked
    pass quadratic in the participant's length.
    """
    f = frame.sort_values(TIME).drop_duplicates(subset=[TIME], keep="last")
    t = pd.to_datetime(f[TIME], errors="coerce")
    f = f.loc[t.notna().to_numpy()]
    return to_ns(f[TIME]), pd.to_numeric(f[CGM], errors="coerce").to_numpy(np.float32)


def _gather(ts, vals, want_ns):
    """Look up want_ns on the exact 5-min grid; unmatched slots become NaN."""
    j = np.searchsorted(ts, want_ns)
    jc = np.clip(j, 0, len(ts) - 1)
    hit = (j < len(ts)) & (ts[jc] == want_ns)
    return np.where(hit, vals[jc], np.nan).astype(np.float32)


def fill_history(hist, mode="interp"):
    """Fill NaNs in (N, HIST_LEN) histories. Returns a NaN-free copy."""
    out = hist.copy()
    if mode == "sentinel":
        return np.nan_to_num(out, nan=-1.0)
    x = np.arange(HIST_LEN)
    for i in range(len(out)):
        row = out[i]
        bad = ~np.isfinite(row)
        if not bad.any():
            continue
        good = ~bad
        row[bad] = np.interp(x[bad], x[good], row[good])  # edges clamp to nearest
    return out


def extract_from_grid(ts, vals, origins_ns, min_valid_frac=0.5, need_targets=False):
    """
    origins_ns : int64 array of origin timestamps (ns).

    Returns (keep_index, hist, targets):
        keep_index : positions in origins_ns that produced a usable window
        hist       : (K, HIST_LEN) raw mg/dL, NaN where unobserved, t last
        targets    : (K, 4) CGM at +30/+60/+90/+120 min, or None
    """
    if len(ts) < 2 or len(origins_ns) == 0:
        return np.empty(0, np.int64), np.empty((0, HIST_LEN), np.float32), None

    lags = np.arange(HIST_LEN - 1, -1, -1, dtype=np.int64) * STEP_NS
    # The lookup builds several int64 (N, 288) temporaries -- ~10x the float32
    # history it returns. Gathering in fixed blocks bounds them; the lookup is
    # pure indexing, so blocking cannot change a value.
    hist = np.empty((len(origins_ns), HIST_LEN), np.float32)
    for s in range(0, len(origins_ns), GATHER_BLOCK):
        o = origins_ns[s : s + GATHER_BLOCK]
        hist[s : s + len(o)] = _gather(ts, vals, o[:, None] - lags[None, :])

    ok = np.isfinite(hist[:, -1]) & (np.isfinite(hist).mean(axis=1) >= min_valid_frac)

    targets = None
    if need_targets:
        leads = np.array(HORIZON_STEPS, dtype=np.int64) * STEP_NS
        targets = _gather(ts, vals, origins_ns[:, None] + leads[None, :])
        ok &= np.isfinite(targets).all(axis=1)

    keep = np.flatnonzero(ok)
    return keep, hist[keep], (targets[keep] if need_targets else None)


def extract(frame, origins_ns, min_valid_frac=0.5, need_targets=False):
    """extract_from_grid for a whole participant frame, grid built here."""
    ts, vals = grid_of(frame)
    return extract_from_grid(ts, vals, origins_ns, min_valid_frac, need_targets)


def candidate_origins(frame, stride_min=15):
    """Origins on the leaderboard's 15-minute cadence, with 24 h of room behind them."""
    ts, _ = grid_of(frame)
    if len(ts) <= HIST_LEN:
        return np.empty(0, np.int64)
    t = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
    on_cadence = (t.minute % stride_min == 0) & (t.second == 0)
    return ts[on_cadence & (ts >= ts[0] + (HIST_LEN - 1) * STEP_NS)]


# --------------------------------------------------------------------------
# Fit set (train.parquet) and eval set (template rows against test.parquet)
# --------------------------------------------------------------------------

def _frame_columns(extra):
    """CGM plus whatever an extra-feature hook reads. The hook never sees CGM."""
    cols = tuple(extra.columns) if extra is not None else ()
    if CGM in cols:
        raise ValueError("an extra-feature hook may not read CGM: past-t CGM is the target")
    return (TIME, CGM) + cols


def build_fit_set(train_path, n_row_groups, per_group, fill="interp", rng=None, verbose=True,
                  extra=None):
    """
    History/target pairs from train.parquet, sampled across the file.

    `extra`, if given, is a hook with a `columns` tuple and a
    `__call__(frame, origins_ns) -> (n, k)` method; its rows for the kept windows
    come back as a fourth array. Window selection is identical with or without it.
    """
    rng = rng or np.random.default_rng(0)
    H, Y, srcs, X = [], [], [], []
    for key, frame in iter_participant_frames(train_path, spread_row_groups(train_path, n_row_groups),
                                              columns=_frame_columns(extra)):
        origins = candidate_origins(frame)
        if len(origins) == 0:
            continue
        if len(origins) > per_group * 4:
            origins = np.sort(rng.choice(origins, size=per_group * 4, replace=False))
        keep, hist, tgt = extract(frame, origins, need_targets=True)
        if len(keep) == 0:
            continue
        kept = origins[keep]
        if len(keep) > per_group:
            sel = np.sort(rng.choice(len(keep), size=per_group, replace=False))
            hist, tgt, kept = hist[sel], tgt[sel], kept[sel]
        H.append(hist)
        Y.append(tgt)
        srcs.extend([key[0]] * len(hist))
        if extra is not None:
            X.append(extra(frame, kept))
        if verbose:
            print(f"    fit {key[0]}/{key[1]}: {len(hist)} windows", flush=True)
    if not H:
        raise RuntimeError(f"No fit windows extracted from {train_path}")
    out = (fill_history(np.concatenate(H), fill), np.concatenate(Y), np.array(srcs))
    return out + (np.concatenate(X),) if extra is not None else out


TEMPLATE_KEY = ["id", "source_file", "date"]


def read_template(path):
    """
    The template's key columns only, with id and source_file as categoricals.

    Read naively, the 2.65 M-row live template is ~430 MB of pandas objects, most
    of it two string columns holding a few hundred distinct values. Read as
    dictionaries it is ~30 MB, and nothing downstream needs more than the keys:
    the pred_* columns are placeholders, and predictions are written fresh.
    """
    return pq.read_table(path, columns=TEMPLATE_KEY,
                         read_dictionary=["id", "source_file"]).to_pandas()


def sample_template_rows(template, n, mode="proportional", seed=0):
    """Sample template rows, keeping the leaderboard's source mix by default."""
    if n is None or n >= len(template):
        return template.copy()
    rng = np.random.default_rng(seed)
    counts = template["source_file"].value_counts()
    weights = pd.Series(1.0, index=counts.index) if mode == "balanced" else counts.astype(float)
    quota = (weights / weights.sum() * n).round().astype(int).clip(upper=counts)
    picks = []
    for src, k in quota.items():
        idx = np.flatnonzero((template["source_file"] == src).to_numpy())
        if k > 0 and len(idx):
            picks.append(rng.choice(idx, size=min(int(k), len(idx)), replace=False))
    return template.iloc[np.sort(np.concatenate(picks))].copy()


EVAL_CHUNK = 50_000


def participant_positions(rows):
    """
    {(source_file, id): ascending int64 positions into `rows`}.

    Grouped on categorical codes rather than by zipping the key columns as Python
    strings: at full-template size the per-row tuples and int lists alone cost
    several hundred MB, and a categorical key never has to be materialised as
    text at all.
    """
    sf = rows["source_file"].astype("category")
    pid = rows["id"].astype("category")
    groups = pd.DataFrame({"sf": sf.cat.codes.to_numpy(), "pid": pid.cat.codes.to_numpy()}) \
        .groupby(["sf", "pid"], sort=False).indices
    sf_names, pid_names = sf.cat.categories, pid.cat.categories
    return {(str(sf_names[a]), str(pid_names[b])): np.asarray(v, np.int64)
            for (a, b), v in groups.items()}


def iter_eval_chunks(test_path, rows, fill="interp", chunk=EVAL_CHUNK, verbose=True, extra=None):
    """
    Yield (index, hist) blocks of at most `chunk` windows for the given template
    rows. `index` are positions into `rows`; rows that never appear produced no
    usable window and are the caller's to fall back for. With an `extra` hook (see
    build_fit_set) the blocks are (index, hist, extra_features).

    This streams rather than returning one array because the full live template is
    2.65 M rows: materialized together the histories alone are ~3 GB and their
    patch embeddings ~24 GB, which no sensible allocation survives. Bounded here,
    peak memory is set by `chunk` and is independent of how many rows are scored,
    so the same code path serves a 20 k laptop subsample and the full set.
    """
    pos_of = participant_positions(rows)
    wanted = set(pos_of)
    rg = row_groups_for(test_path, wanted)
    if verbose:
        total = pq.ParquetFile(test_path).metadata.num_row_groups
        print(f"    reading {len(rg)}/{total} row groups for {len(wanted)} participants", flush=True)
    origin_ns = to_ns(rows[TIME])

    buf_idx, buf_hist, buf_x, buffered = [], [], [], 0

    def flush():
        idx = np.concatenate(buf_idx)
        hist = fill_history(np.concatenate(buf_hist), fill)
        x = np.concatenate(buf_x) if extra is not None else None
        buf_idx.clear()
        buf_hist.clear()
        buf_x.clear()
        return (idx, hist, x) if extra is not None else (idx, hist)

    for key, frame in iter_participant_frames(test_path, rg, columns=_frame_columns(extra),
                                              wanted=wanted):
        where = pos_of.get(key, np.empty(0, np.int64))
        if len(where) == 0:
            continue
        ts, vals = grid_of(frame)          # once per participant, not once per chunk
        kept = 0
        for start in range(0, len(where), chunk):
            sl = where[start : start + chunk]
            keep, hist, _ = extract_from_grid(ts, vals, origin_ns[sl], need_targets=False)
            if len(keep) == 0:
                continue
            buf_idx.append(sl[keep])
            buf_hist.append(hist)
            if extra is not None:
                buf_x.append(extra(frame, origin_ns[sl[keep]]))
            buffered += len(keep)
            kept += len(keep)
            if buffered >= chunk:
                yield flush()
                buffered = 0
        if verbose:
            print(f"    eval {key[0]}/{key[1]}: {kept}/{len(where)} rows windowed", flush=True)

    if buffered:
        yield flush()
