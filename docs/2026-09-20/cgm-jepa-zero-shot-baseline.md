# CGM-JEPA zero-shot-encoder baseline, and repo conventions

## 1. What changed

**New conventions files**

- `CLAUDE.md` — documentation policy, deprecation policy, suite cheat sheet,
  project state, next steps. Adds a repo-specific fourth rule: `run.py`,
  `metrics.py`, and the `data/*/template.parquet` + `targets.parquet` files are
  competition-supplied and must not be modified.
- `EXPERIMENTS.md` — current-only spec with a §0 divergence register.
- `docs/YYYY-MM-DD/` — this file is the first entry.

**New modules** (all additive; nothing existing was touched)

- `jepa_model.py` — CGM-JEPA `Encoder`, `DataEmbedding`, `ValueEmbedding`,
  `Block`, `MultiHeadAttention`, `MLP`, plus `load_encoder` and
  `embed_windows`. Loads `CRUISEResearchGroup/CGM-JEPA` safetensors directly.
- `jepa_windows.py` — parquet → 288-sample history / 4-horizon target pairs.
  Row-group statistics selection, participant-block iteration, exact-grid
  gather, gap fill, fit-set and eval-set builders, stratified template sampling.
- `predict_jepa.py` — driver. Required `--date`, fail-fast input validation,
  frozen encoder → ridge head → predictions + append-only results row.
- `requirements-jepa.txt` — `torch`, `safetensors`, `huggingface_hub`.

**New directories**

- `weights/cgm_jepa_hf/{cgm_jepa,x_cgm_jepa}/` — published checkpoints, 4 MB each.
- `results/<date>/<experiment>/` and `logs/<date>/<experiment>/`.

**Unchanged:** `run.py`, `metrics.py`, `model.py`, `train.py`,
`prepare_windows.py`, `requirements.txt`, `README.md`, `data/README.md`,
everything under `data/`.

## 2. Why

Victor's ask: *"implement this cgm-jepa paper's model, using their weights and
run a preliminary zero-shot prediction on the challenge data with zero
finetuning and zero training [...] follow the code structure in the codebase and
not change any structure code as this is what the competition wants us to format
in."*

Surveying the paper and the authors' repo first turned up a blocker worth
stating plainly, because it determines the shape of everything below.

**CGM-JEPA cannot forecast glucose zero-shot. Not as implemented here — at all.**

- The pretraining objective (`pretrain/pretrain_cgm_jepa.py`) is
  `mean |predictor(context) − layernorm(EMA_encoder(x))|`, computed entirely in
  96-d latent space. Nothing in the loss touches mg/dL.
- The architecture has no decoder. The encoder's only non-trunk head, `proj`
  (96 → 1024 → 48), is on neither the CGM nor the Glucodensity loss path in
  either pretraining script; it is dead weight in the checkpoint.
- `save_model("cgm_jepa", encoder, path_save)` saves the **encoder only**. The
  Hugging Face repo confirms it: `cgm_jepa/model.safetensors`,
  `x_cgm_jepa/model.safetensors`, plus two unrelated baselines. The JEPA
  `Predictor` was never released, so even latent-space rollout is unavailable.

So a readout on frozen embeddings is not a shortcut — it is the only path from
these weights to a number in mg/dL. Victor, told this: *"i dont really need the
'literal zero-training forecast' i just want something thats easy to implement
and reliable as an initial baseline."* Scope: *"write a small subsample test for
my laptop. i will give you instructions later for running it on a cluster."*

## 3. Decisions made

- **Readout = closed-form ridge, not k-NN retrieval or a trained MLP.** A single
  `np.linalg.solve` has no optimizer, no schedule, and no seed sensitivity worth
  arguing about; it is the paper's own frozen-encoder linear-probe protocol moved
  from classification to regression. Retrieval in embedding space was the other
  candidate and would have honoured "zero training" literally, but it needs a
  reference bank and a neighbourhood size, i.e. more knobs for a *baseline*.
- **Target is Δ from current glucose, not the level.** `G(t+h) = G(t) + wφ(x)`
  makes persistence the zero solution, so a head that learns nothing degrades to
  persistence rather than to noise, and the JEPA-vs-persistence gap in the
  results file is a direct readout of what the representation contributed.
  Predicting the level would have let the head spend its capacity re-deriving
  "glucose is near where it is now" and hidden that signal.
- **Features = last-patch ⊕ mean-pooled embedding (192-d).** The paper's
  downstream protocol mean-pools all 24 patches, which is right for a day-level
  phenotype and wrong for a 30-minute forecast: it discards recency. Keeping the
  last patch restores it. Nothing hand-crafted is mixed in — no raw CGM lags, no
  trend features — so the comparison measures the encoder, not our feature
  engineering.
- **Raw mg/dL inputs, no time features.** Pretraining ran with
  `normalize_x: False` and `use_time_feature: False`. Z-scoring the input or
  switching the time embedding on would push inputs off the distribution the
  weights were fitted on. The time-feature module is kept in the port only so
  state dicts load strictly.
- **Interior gaps interpolated, not sentinel-filled.** The authors' transformer
  uses `-1` for tail padding; inside a 100–300 mg/dL series a `-1` is an extreme
  outlier to a raw-value input convolution. `--fill sentinel` reproduces their
  convention for comparison.
- **λ selected on a participant-disjoint holdout.** 24 h windows on a 15-minute
  stride overlap by ~97%, so a random split would tune the penalty against near
  copies of the fitting rows.
- **`x_cgm_jepa` is the default encoder.** It ranks first or second on every
  endpoint-regime cell in the paper; `cgm_jepa` is selectable for an A/B.
- **Persistence is logged every run, on the same rows, from the same `G(t)`.**
  Not a separate script — a separate script could drift in row set or windowing,
  and then the comparison would measure the pipeline instead of the method.
- **Subsample defaults to `proportional`.** The live test set is 77% `Loop`; a
  balanced sample would report a number that does not predict the full-set
  result. `--sample-mode balanced` exists for per-source diagnostics.
- **Conventions installed as-is, except the results layout for `model.py`.**
  `train.py` writes `checkpoints/gluco_v1.pt` and prints to stdout rather than
  using `results/<date>/`. Not migrated: that family has no result files yet, so
  the migration is free later, and result files are never moved.

**Verification performed.** The port reproduces the authors' `models/encoder.py`
exactly — maximum absolute difference **0.0** on both checkpoints, on token
output, `proj` output, and the mean-pooled embedding. The full driver was then
run end-to-end against a synthetic fixture built to the real parquet schema
(three source datasets, nine participants, 5-minute grid, injected gaps), and
its `--scope full` output passes the competition's own `run.py --competition
live --horizon all` format validation.

Two real bugs were caught by that run and fixed:

- **Timestamp unit.** `pd.to_datetime(...).to_numpy().astype("int64")` returns
  *microseconds* under pandas ≥ 3, not nanoseconds. Every exact-grid lookup
  missed and the extractor silently produced zero windows — no error, no warning.
  Pinned via `jepa_windows.to_ns`. This would not have shown on the laptop
  (pandas 2.3.3) and would have broken on a cluster with a newer pandas.
- **Carry across non-adjacent row groups.** When the caller selects a spread of
  row groups, the partial participant block held from row group *i* was being
  concatenated onto row group *j > i+1*, which could join two disjoint stretches
  of one participant into a single frame. Now flushed as its own block.

**First real-data run.** `--scope sub --eval-rows 20000` on the live set,
19,999 rows sampled proportionally, 13,761 fit windows from 6 source datasets,
19.2 s wall clock on the laptop. RMSE mg/dL:

| Horizon | JEPA+ridge | Persistence | Δ |
|---|---|---|---|
| 30 min | 21.17 | 25.21 | −4.04 |
| 60 min | 34.90 | 39.72 | −4.82 |
| 90 min | 42.36 | 48.87 | −6.51 |
| 120 min | 46.57 | 55.19 | −8.62 |

The margin widens with horizon, which is the signature of an embedding carrying
trajectory information rather than restating the current level — and it partly
answers the forecast-blindness worry below: the representation is not inert,
even if the synthetic flat-vs-rising probe made it look close to it.

## 4. What did NOT change / out of scope

- **No submission was produced and nothing was submitted.** `--scope full` has
  never been run on the real data.
- **`model.py` / `train.py` / `prepare_windows.py` untouched and still unrun.**
  No `checkpoints/`, no `model_data_v1/`.
- **The annual competition is untouched.** Its template is missing locally and
  its input dataset is behind a sign-in; the `-2` / `-1` masking convention is
  unhandled (divergence register §0, item 2).
- **`subject_split_across_traintest` is not filtered.** The ridge head is fitted
  on `train.parquet` participants with no check that they are absent from the
  evaluation rows.
- **The team model's insulin/basal/bolus/carbs channels are unused here.** The
  published encoder is CGM-only, so this baseline cannot be a "what-if" model.
- **`README.md` not rewritten.** It is upstream's and describes the submission
  contract correctly.

## 5. Follow-ups

- **Run `--scope full`** and validate with `run.py` before any submission.
  `--scope sub` is done; the full pass is the remaining gate.
- **Filter or report on `subject_split_across_traintest`.** Cheapest version:
  report the subsample metric split by the flag. Proper version: exclude those
  participants from the fit set.
- **A/B `cgm_jepa` vs `x_cgm_jepa`, and `--fill interp` vs `sentinel`.** Four
  cheap runs; the tag grammar already separates them in one results file.
- **Worth watching: the encoder may be nearly forecast-blind.** A flat 120 mg/dL
  day and a monotone 80→260 day embed to a cosine similarity of 1.0000 and an L2
  distance of 0.08 against a mean token magnitude of ~0.88. If the real-data run
  shows the ridge barely beating persistence, that is the reason, and it is a
  property of the pretrained representation rather than a bug in this code.
- **Cluster port.** Inference is CPU-only and single-process. The knobs to raise
  are `--eval-rows`, `--fit-row-groups`, `--fit-per-group`, `--batch-size`.
  `weights/` must be synced or the hub reachable. Exclude `.pylibs/`,
  `.venv-linux/`, `.claude-tmp/` from any rsync — an `--exclude` list does not
  read `.gitignore`.
- **Local scaffolding to clean up.** `.pylibs/`, `.venv-linux/`,
  `.venv-linux-tmp/`, `.claude-tmp/`, `.exectest.sh`, `.symtest` were created in
  the repo root to get a Linux toolchain onto the mounted folder. All are
  gitignored; none are needed once a native environment exists.
