# UNITE — Unofficial Reproduction

> **Note**: This is an **unofficial reproduction** of the UNITE paper. It is not the original authors' implementation. For the original paper, please refer to the citation below.

This repository reproduces the training pipeline of **UNITE** from the paper:

> **Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content**
>
> Rohit Kundu, Hao Xiong, Vishal Mohanty, Athula Balachandran, Amit K. Roy-Chowdhury
>
> CVPR 2025

The current implementation supports binary classification (`REAL → 0`, `FAKE → 1`) trained on **FaceForensics++ C23** only (no SAIL-VOS-3D/GTA-V). It runs the full train → validate → test loop but is **not** the complete `FF++ + GTA-V` universal detector setup from the paper.

## Directory Structure

```text
./
├── download_FFpp.py                       # Download & relocate FF++ C23 data
├── requirements.txt                       # Python dependencies
├── configs/
│   └── unite_ffpp_c23.yaml                # Default training config
├── scripts/
│   └── train_ddp.sh                       # DDP launcher script (edit vars at top)
├── train.py                               # Training entry point
├── eval.py                                # Evaluation entry point
├── src/
│   ├── data/                              # CSV, video reader, Dataset
│   ├── engine/                            # Training & evaluation loop
│   ├── losses/                            # CE + Attention Diversity loss
│   ├── models/                            # SigLIP encoder + UNITE Transformer
│   └── utils/                             # Config, metrics, checkpoint, logging
└── data/
    ├── build_ffpp_splits.py               # Build FF++ train/val/test splits
    └── FaceForensics++_C23/               # FF++ C23 data root
```

Training outputs are automatically organized under `./outputs/` by key hyperparameters, e.g.:

```text
outputs/ffpp_c23/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0.5/lr0.0001_effbs32_gc1_amp0/seed42/
```

Use `--run_name <name>` to override the last path segment (e.g. `seed42` → `smoke`).

## 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If your CUDA / PyTorch environment is incompatible with the versions pinned in `requirements.txt`, install a matching PyTorch build for your CUDA version first, then install the remaining dependencies.

## 2. Prepare FF++ C23 Data

The download script uses KaggleHub to fetch `xdxd003/ff-c23` and places it under:

```text
data/FaceForensics++_C23
```

Run:

```bash
python download_FFpp.py
```

Expected directory structure:

```text
./data/FaceForensics++_C23
├── DeepFakeDetection/
├── Deepfakes/
├── Face2Face/
├── FaceShifter/
├── FaceSwap/
├── NeuralTextures/
├── original/
└── csv/
    ├── FF++_Metadata.csv
    ├── DeepFakeDetection.csv
    ├── Deepfakes.csv
    ├── Face2Face.csv
    ├── FaceShifter.csv
    ├── FaceSwap.csv
    ├── NeuralTextures.csv
    └── original.csv
```

The master metadata file is:

```text
./data/FaceForensics++_C23/csv/FF++_Metadata.csv
```

The `File Path` column contains paths relative to the `FaceForensics++_C23` data root, e.g.:

```text
DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4
```

## 3. Build Train / Val / Test Splits

FF++ serves as both the training source domain and the in-domain evaluation set. To prevent data leakage, split it into three subsets:

- `train` — model training
- `val` — checkpoint selection (`best_auc.ckpt`) and hyperparameter tuning
- `test` — final in-domain (FF++) evaluation

Run after downloading:

```bash
python data/build_ffpp_splits.py \
  --input_csv data/FaceForensics++_C23/csv/FF++_Metadata.csv \
  --data_root data/FaceForensics++_C23 \
  --out_dir data/FaceForensics++_C23/splits \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --require_exists
```

Output files:

```text
data/FaceForensics++_C23/splits/train.csv
data/FaceForensics++_C23/splits/val.csv
data/FaceForensics++_C23/splits/test.csv
```

`--require_exists` filters out videos that haven't finished downloading or have mismatched paths. An empty split usually means the data isn't fully downloaded yet, or the relative paths in the CSV don't match the actual directory layout.

## 4. Speeding Up Training

The main bottlenecks are **SigLIP encoding** (64 frames × 384×384 per sample through a frozen model) and **gradient accumulation** (single-GPU `batch_size=1` needs 32 steps to assemble one effective batch). The following techniques are listed in descending order of impact.

### 4.1 Multi-GPU DDP (largest gain)

With DDP, gradient accumulation steps are automatically scaled down by the number of GPUs while keeping the global effective batch size at 32. E.g., 4 GPUs reduce `gradient_accumulation_steps` from 32 → 8, yielding ~4× fewer effective training steps.

```bash
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --config configs/unite_ffpp_c23.yaml --run_name exp_ddp

# 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
  --config configs/unite_ffpp_c23.yaml --run_name exp_ddp
```

Or edit `GPUS` and `NPROC` in `scripts/train_ddp.sh` and run:

```bash
bash scripts/train_ddp.sh
```

### 4.2 Enable AMP (Automatic Mixed Precision)

Set `train.amp: true` in the config. Reduces memory and compute time by ~30–40%. SigLIP weights remain frozen in fp32, but the transformer forward/backward uses fp16 — safe and noticeably faster on most GPUs.

```yaml
train:
  amp: true
```

### 4.3 Increase encoder_batch_size

`model.encoder_batch_size` controls how many frames SigLIP processes per forward call. Default is 4; raise to 8 or 16 if VRAM allows to significantly reduce encoder calls.

```yaml
model:
  encoder_batch_size: 8   # or 16, depending on VRAM
```

### 4.4 Increase DataLoader Parallelism

- **`num_workers`** — 4–8 on Linux; keep at 0 on Windows (multi-processing is unstable on Windows).
- **`prefetch_factor`** — batches pre-loaded per worker; default 2, can raise to 4.
- **`pin_memory: true`** (default on) — speeds up CPU→GPU transfer.

```yaml
data:
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
```

### 4.5 Reduce I/O & Validation Overhead

- **`log_every`** — writes CSV/TensorBoard every N steps (default 10); raise to 50–100 for small disk savings.
- **`validate_every_epoch: false`** — skip per-epoch validation; evaluate only at the end.
- **`save_every_epoch: false`** — reduce checkpoint writes (saves disk I/O, doesn't affect training speed).
- **`debug_anomaly: false`** (default off) — enables PyTorch anomaly detection; drastically slows training. Only for NaN debugging.
- **`log_timing: false`** (default off) — inserts `cuda.synchronize`; slows training. Only for profiling.

### 4.6 Fewer Epochs / Quick Smoke Test

For a fast pipeline check, limit steps via CLI arguments (no config changes needed):

```bash
python train.py --config configs/unite_ffpp_c23.yaml --device cuda:0 \
  --max_train_steps 20 --max_val_steps 5
```

If convergence is fast enough, you can also reduce `train.epochs` (default 25) to 15–20.

### 4.7 Quick Single-GPU Tuning Reference

Fastest single-GPU config (assuming ≥24 GB VRAM):

```yaml
data:
  num_workers: 8         # Linux; use 0 on Windows

model:
  encoder_batch_size: 8  # lower to 4 if OOM

train:
  amp: true              # mixed precision
  log_every: 50
  validate_every_epoch: true
```

## 5. Configuration Reference

Default config: `configs/unite_ffpp_c23.yaml`. Non-obvious parameters explained below.

### data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temporal_stride` | 2 | Frame sampling stride; 1 = consecutive, 2 = every other frame |
| `train_random_start` | true | Random start position for 64-frame clips during training |
| `require_exists` | true | Filter out video files that don't exist on disk |
| `num_workers` | 0 | DataLoader workers; keep at 0 on Windows |

### model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `siglip_model_name` | google/siglip-so400m-patch14-384 | HuggingFace model ID or local path |
| `local_files_only` | false | true = use only cached weights |
| `freeze_siglip` | true | Freeze the vision encoder (no gradient updates) |
| `encoder_batch_size` | 4 | Max frames per SigLIP forward call; lower if OOM |
| `transformer_depth` | 4 | Number of transformer encoder layers |
| `num_heads` | 12 | Multi-head attention heads |
| `mlp_ratio` | 4.0 | MLP hidden dim = hidden_size × ratio |
| `dropout` | 0.1 | Dropout rate in transformer |

### loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ce_weight` / `ad_weight` | 0.5 | Weights for CE and Attention Diversity losses |
| `center_eta` | 0.05 | EMA update rate for class centers in AD loss |
| `delta_between` | 0.5 | Penalty threshold for inter-class center distance |
| `delta_within` | [0.01, -2.0] | Within-class distance margins (real / fake) |

### train

| Parameter | Default | Description |
|-----------|---------|-------------|
| `effective_batch_size` | 32 | Target global batch size (documentation; not directly used) |
| `gradient_accumulation_steps` | 32 | Gradient accumulation steps; auto-scaled ÷ world_size in DDP |
| `scheduler_step_size` | 1000 | LR decay every N optimizer steps |
| `scheduler_gamma` | 0.5 | LR decay factor |
| `amp` | false | Automatic mixed precision |
| `grad_clip_max_norm` | 1.0 | Global gradient norm clipping threshold |
| `warmup_steps` | 0 | Linear LR warmup steps; 0 = off |
| `debug_anomaly` | false | PyTorch anomaly detection (for NaN debugging only) |
| `log_grad_norm` | false | Print gradient norm to console each optimizer step |
| `log_every` | 10 | Write CSV metrics every N optimizer steps |
| `validate_every_epoch` | true | Run validation after each epoch |

## 6. Smoke Test

Run a short test to verify that data loading, model, loss, and checkpointing all work.

Single GPU:

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --max_train_steps 20 \
  --max_val_steps 5
```

Multi-GPU (DDP):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config configs/unite_ffpp_c23.yaml \
  --max_train_steps 20 \
  --max_val_steps 5
```

If you hit OOM, try in order:

1. Keep `train.batch_size: 1`
2. Lower `model.encoder_batch_size`
3. Maintain or increase `train.gradient_accumulation_steps`

## 7. Full Training

### Single GPU

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01
```

### Multi-GPU (DDP)

GPUs are not auto-detected — you must explicitly list them via `CUDA_VISIBLE_DEVICES`:

```bash
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/unite_ffpp_c23.yaml --run_name exp01

# 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --config configs/unite_ffpp_c23.yaml --run_name exp01
```

Under DDP, `gradient_accumulation_steps` is automatically scaled down by the number of GPUs to keep the global effective batch size at 32. E.g., accum=8 for 4 GPUs, accum=4 for 8 GPUs.

Or use the launcher script (edit parameters inside the script as needed):

```bash
bash scripts/train_ddp.sh
```

During training, the following artifacts are saved under the resolved experiment directory:

```text
<run_dir>/latest_train.ckpt
<run_dir>/last.ckpt
<run_dir>/best_auc.ckpt
<run_dir>/metrics_train.csv
<run_dir>/metrics_val.csv
<run_dir>/tensorboard/
```

The actual path is printed at startup: `Output directory: <run_dir>`.

- `latest_train.ckpt` — resume point after train completes but before validation; resume skips directly to validation
- `last.ckpt` — full epoch checkpoint (train + validation + save all done)
- `best_auc.ckpt` — best checkpoint by FF++ validation ROC AUC
- `metrics_train.csv` — training metrics log (loss, center distances, etc.)
- `metrics_val.csv` — validation metrics log (ROC AUC, PR AUC, accuracy, etc.)
- `tensorboard/` — TensorBoard event files

View TensorBoard:

```bash
tensorboard --logdir <run_dir>/tensorboard
```

## 8. Resume Training

Recommended: resume from a completed epoch checkpoint:

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01 \
  --resume <run_dir>/last.ckpt
```

If training was interrupted during validation, `latest_train.ckpt` skips the already-completed training and only runs the pending validation for that epoch, ensuring the validation curve in TensorBoard has no gaps:

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01 \
  --resume <run_dir>/latest_train.ckpt
```

When using scripts, edit `RUN_NAME` and `RESUME` in `scripts/train_ddp.sh`. The `RUN_NAME` should match the original training run. TensorBoard logs write to the same `<run_dir>/tensorboard/` directory; on resume, `writer_step` and `global_step` are restored from the checkpoint, so training loss curves append continuously by `writer_step`, learning rate / gradient norm by `global_step`, and validation curves fill in by epoch.

Only load checkpoints you trained yourself. For compatibility with PyTorch 2.6+ `torch.load` safety defaults, local checkpoints are loaded with `weights_only=False`.

## 9. Evaluation

Validation set:

```bash
python eval.py \
  --config configs/unite_ffpp_c23.yaml \
  --ckpt <run_dir>/best_auc.ckpt \
  --split val \
  --device cuda
```

Test set:

```bash
python eval.py \
  --config configs/unite_ffpp_c23.yaml \
  --ckpt <run_dir>/best_auc.ckpt \
  --split test \
  --device cuda
```

Evaluation is clip-level binary classification. Output metrics include:

- accuracy
- ROC AUC
- PR AUC / average precision
- precision@0.5
- recall@0.5
- precision@recall=0.8
- recall@precision=0.8

## 10. Cross-Domain Evaluation

The paper's core goal is universal detection — don't rely solely on FF++ in-domain results. After FF++-only training, prepare external datasets for cross-domain evaluation.

Suitable cross-domain datasets from the paper:

- Face manipulation: CelebDF, DeeperForensics, Deepfake-TIMIT, HifiFace, UADFV
- Background manipulation: AVID
- Fully synthetic: DeMamba; if unused during training, GTA-V can also serve as fully synthetic test data
- In-the-wild: NYTimes DeepFake Quiz

This repository does not yet include download and format conversion scripts for these external datasets. When adding them, maintain CSV columns consistent with the FF++ splits:

```text
abs_path,rel_path,label
```

Then reuse `eval.py` for evaluation. Cross-domain test sets must not overlap with FF++ training or validation splits.

## 11. GTA-Free Variant (Innovation)

The GTA-free variant adds three modules on top of the baseline to mitigate face-only bias and improve cross-domain generalization:

- **Face masking augmentation** — OpenCV Haar cascade constructs face-masked views during training
- **Attention anti-collapse loss** — prevents attention from collapsing onto too few frames
- **Counterfactual consistency loss** — enforces prediction consistency between original and face-masked views
- **Class-balanced CE** — inverse-frequency weights to mitigate fake-biased predictions

Use the dedicated config:

```bash
# baseline
python train.py --config configs/unite_ffpp_c23.yaml --device cuda:0 \
  --run_name baseline_exp01

# GTA-free variant
python train.py --config configs/unite_ffpp_c23_gta_free.yaml --device cuda:0 \
  --run_name gta_free_exp01
```

For DDP, change `CONFIG` and `RUN_NAME` in `scripts/train_ddp.sh`. Or use the variant-specific launcher:

```bash
bash scripts/train_gta_free.sh
```

The GTA-free output directory includes a `variant=gta_free` layer, fully isolated from the `baseline` layer.

## 12. FAQ

### Empty split or missing videos

The data likely hasn't finished downloading, or the `File Path` in the CSV doesn't match the actual data root. Verify videos are under:

```text
data/FaceForensics++_C23
```

Then re-run the split-building command.

### `ModuleNotFoundError: No module named 'torch'`

Install dependencies:

```bash
pip install -r requirements.txt
```

If the PyTorch version is incompatible with your CUDA version, reinstall PyTorch matching your CUDA environment.

### HuggingFace SigLIP weight download fails

The default model is:

```text
google/siglip-so400m-patch14-384
```

If the training machine lacks internet access, pre-download and cache the model, or change `model.siglip_model_name` in `configs/unite_ffpp_c23.yaml` to a local path.

### Out of memory (OOM)

Priority adjustments:

```yaml
train:
  batch_size: 1
  gradient_accumulation_steps: 32

model:
  encoder_batch_size: 1
```

SigLIP encoding of 64-frame clips is expensive. Start with a small smoke test on single GPU.

### DDP `Address already in use` or port conflict

`torchrun` uses a random port by default. If there's a conflict, specify one explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 train.py --config configs/unite_ffpp_c23.yaml
```

### Higher per-GPU memory usage under DDP vs. single GPU

DDP performs all-reduce communication during backward, which consumes a small amount of extra memory. If you hit OOM, reduce per-GPU batch size or keep `batch_size: 1`. Multi-GPU training already speeds things up by reducing accumulation steps — you don't need to increase batch size.

## Citation

If you use this code or find it helpful, please cite the original UNITE paper:

```bibtex
@inproceedings{kundu2025unite,
  author    = {Kundu, Rohit and Xiong, Hao and Mohanty, Vishal and Balachandran, Athula and Roy-Chowdhury, Amit K.},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title     = {Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content},
  year      = {2025},
  pages     = {28050-28060}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
