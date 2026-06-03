# UNITE 训练流程复现（非官方实现）

> **注意**：本项目是论文 **Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content** 的**非官方复现代码**，并非作者发布的原始实现。如需引用，请参考文末的论文 BibTeX。

本项目用于复现上述论文中 UNITE 的训练流程。当前实现优先支持在 **FaceForensics++ C23** 上进行 FF++ only 的二分类训练：`REAL -> 0`，`FAKE -> 1`。

当前版本不包含 SAIL-VOS-3D/GTA-V 训练数据，因此可以跑通 UNITE 的训练、验证、测试流程，但不是论文完整的 `FF++ + GTA-V` universal detector 设置。

## 目录结构

```text
./
├── download_FFpp.py                       # 下载并移动 FF++ C23 数据
├── requirements.txt                       # Python 依赖
├── configs/
│   └── unite_ffpp_c23.yaml                # 默认训练配置
├── scripts/
│   └── train_ddp.sh                       # 训练启动脚本（可编辑参数）
├── train.py                               # 训练入口
├── eval.py                                # 评估入口
├── src/
│   ├── data/                              # CSV、视频读取、Dataset
│   ├── engine/                            # 训练与评估循环
│   ├── losses/                            # CE + Attention Diversity loss
│   ├── models/                            # SigLIP encoder + UNITE Transformer
│   └── utils/                             # 配置、指标、checkpoint、日志
└── data/
    ├── build_ffpp_splits.py               # 构建 FF++ train/val/test split
    └── FaceForensics++_C23/               # FF++ C23 数据根目录
```

训练输出会按关键参数自动分层写到 `./outputs/` 下，例如：

```text
outputs/ffpp_c23/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0.5/lr0.0001_effbs32_gc1_amp0/seed42/
```

如果指定 `--run_name smoke`，最后一级目录会从 `seed42` 变成 `smoke`。

## 1. 安装环境

建议先创建并激活虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

如果你的 CUDA / PyTorch 环境和 `requirements.txt` 中的版本不兼容，请按本机 CUDA 版本安装对应的 PyTorch，再安装其余依赖。

## 2. 准备 FF++ C23 数据

数据下载脚本会使用 KaggleHub 下载 `xdxd003/ff-c23`，并移动到：

```text
data/FaceForensics++_C23
```

运行：

```bash

python download_FFpp.py
```

预期目录结构：

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

主元数据文件是：

```text
./data/FaceForensics++_C23/csv/FF++_Metadata.csv
```

其中 `File Path` 是相对 `FaceForensics++_C23` 数据根目录的路径，例如：

```text
DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4
```

## 3. 构建 train / val / test split

FF++ 既是训练源域，也可以作为域内评估数据。为了避免数据泄漏，需要拆成三份：

- `train`：训练模型
- `val`：选择 `best_auc.ckpt` 和调参
- `test`：最终报告 FF++ 域内测试结果

数据下载完成后运行：

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

输出文件：

```text
data/FaceForensics++_C23/splits/train.csv
data/FaceForensics++_C23/splits/val.csv
data/FaceForensics++_C23/splits/test.csv
```

`--require_exists` 会过滤还没有下载完成或路径不匹配的视频。如果 split 为空，通常说明数据还没有下载完成，或 CSV 中的相对路径与实际目录不一致。

## 4. 加快训练速度

训练的主要瓶颈来自 **SigLIP 编码**（每样本 64 帧 × 384×384 逐一过 frozen 模型）和 **梯度累积**（单卡 batch_size=1，需 32 步才凑出一个 effective batch）。以下按收益从大到小列出加速手段。

### 4.1 多卡 DDP（收益最大）

多卡并行训练时梯度累积步数自动按卡数缩减，保持全局 effective batch=32 不变。例如 4 卡时 `gradient_accumulation_steps` 自动从 32 → 8，等效训练步数减少 4 倍。

```bash
# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --config configs/unite_ffpp_c23.yaml --run_name exp_ddp

# 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
  --config configs/unite_ffpp_c23.yaml --run_name exp_ddp
```

或直接编辑 `scripts/train_ddp.sh` 中的 `GPUS` 和 `NPROC` 后运行：
```bash
bash scripts/train_ddp.sh
```

### 4.2 开启混合精度（AMP）

在配置中将 `train.amp` 设为 `true`，可减少约 30-40% 显存和计算时间。虽然 SigLIP 权重冻结且以 fp32 运行，但 transformer 部分的 forward/backward 会使用 fp16，在大多数 GPU 上安全且明显提速。

```yaml
train:
  amp: true
```

或在已有配置上临时覆盖（修改 yaml 或直接用脚本传参不方便时建议直接改 yaml）。

### 4.3 提高 encoder_batch_size

`model.encoder_batch_size` 控制 SigLIP 一次前向处理多少帧。默认 4，显存充裕时可提高到 8 或 16，显著减少 encoder 调用次数。

```yaml
model:
  encoder_batch_size: 8   # 或 16，视显存而定
```

### 4.4 增大 DataLoader 并行度

- **`num_workers`**：Linux 下建议设为 4-8；Windows 下建议保持 0（多进程在 Windows 上不稳定）。
- **`prefetch_factor`**：每个 worker 预取的 batch 数，默认 2，可提高到 4。
- **`pin_memory: true`**（默认开启）：配合 GPU 训练时加速 CPU→GPU 传输。

```yaml
data:
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
```

### 4.5 减少 Io 与验证开销

- **`log_every`**：默认每 10 步写一次 CSV/tensorboard，调大到 50 或 100 可减少少量磁盘开销。
- **`validate_every_epoch: false`**：快速实验时可关闭逐 epoch 验证，只在最后评估。
- **`save_every_epoch: false`**：减少 checkpoint 写入（不影响训练速度，但节省磁盘 I/O）。
- **`debug_anomaly: false`**（默认关闭）：开启会大幅拖慢训练，仅调试 NaN 时才用。
- **`log_timing: false`**（默认关闭）：开启后会插入 `cuda.synchronize`，拖慢训练，仅 profile 时用。

### 4.6 减少 epoch 数 / 快速 smoke test

快速验证流程是否跑通时，用命令行参数限制步数，无需改配置：

```bash
python train.py --config configs/unite_ffpp_c23.yaml --device cuda:0 \
  --max_train_steps 20 --max_val_steps 5
```

正式训练如果收敛足够快，也可以减少 `train.epochs`（默认 25）到 15-20。

### 4.7 单卡调参速查

单卡下最快的配置组合（按显存 24GB+ 调整）：

```yaml
data:
  num_workers: 8         # Linux；Windows 用 0

model:
  encoder_batch_size: 8  # 显存不足则降回 4

train:
  amp: true              # 混合精度
  log_every: 50
  validate_every_epoch: true
```

## 5. 配置文件

默认配置位于 `configs/unite_ffpp_c23.yaml`。以下说明非自明参数：

### data

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temporal_stride` | 2 | 隔帧采样步长，1=逐帧，2=每隔一帧 |
| `train_random_start` | true | 训练时在视频中随机位置取 64 帧片段 |
| `require_exists` | true | 过滤尚不存在的视频文件 |
| `num_workers` | 0 | DataLoader 进程数，Windows 建议 0 |

### model

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `siglip_model_name` | google/siglip-so400m-patch14-384 | HuggingFace 模型标识或本地路径 |
| `local_files_only` | false | true=仅使用本地缓存的权重 |
| `freeze_siglip` | true | 冻结视觉编码器，不参与梯度更新 |
| `encoder_batch_size` | 4 | SigLIP 每次编码的帧数上限，显存不足时降低 |
| `transformer_depth` | 4 | Transformer encoder 层数 |
| `num_heads` | 12 | 多头注意力头数 |
| `mlp_ratio` | 4.0 | MLP 隐藏层维度 = hidden_size × ratio |
| `dropout` | 0.1 | Transformer 中 dropout 比例 |

### loss

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ce_weight` / `ad_weight` | 0.5 | CE 和 Attention Diversity 损失的权重 |
| `center_eta` | 0.05 | AD loss 中类别中心的 EMA 更新速率 |
| `delta_between` | 0.5 | 类间中心距离低于此值产生 penalty |
| `delta_within` | [0.01, -2.0] | 各类别的类内距离 margin（real / fake） |

### train

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `effective_batch_size` | 32 | 目标全局 batch size（文档值，不直接使用） |
| `gradient_accumulation_steps` | 32 | 梯度累积步数；DDP 时自动 ÷ 卡数 |
| `scheduler_step_size` | 1000 | 每 N 个 optimizer step 衰减一次学习率 |
| `scheduler_gamma` | 0.5 | 学习率衰减因子 |
| `amp` | false | 混合精度训练；SigLIP 冻结时不推荐开 |
| `grad_clip_max_norm` | 1.0 | 梯度总范数裁剪阈值 |
| `warmup_steps` | 0 | LR 线性预热步数，0=关闭 |
| `debug_anomaly` | false | PyTorch 自动异常检测，定位 NaN 用 |
| `log_grad_norm` | false | 每个 optimizer step 打印梯度范数到终端 |
| `log_every` | 10 | 每 N 个 optimizer step 写一次 CSV 指标 |
| `validate_every_epoch` | true | 每个 epoch 后验证 |

## 6. Smoke test

先跑一个很短的测试，确认数据、模型、loss、checkpoint 流程能通。

单卡：

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --max_train_steps 20 \
  --max_val_steps 5
```

多卡 (DDP)：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config configs/unite_ffpp_c23.yaml \
  --max_train_steps 20 \
  --max_val_steps 5
```

如果显存不足，优先尝试：

1. 保持 `train.batch_size: 1`
2. 降低 `model.encoder_batch_size`
3. 保持或增加 `train.gradient_accumulation_steps`

## 7. 正式训练

### 单卡

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01
```

### 多卡 (DDP)

程序不会自动探测 GPU，你需要通过 `CUDA_VISIBLE_DEVICES` 显式指定用哪些卡：

```bash
# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/unite_ffpp_c23.yaml --run_name exp01

# 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --config configs/unite_ffpp_c23.yaml --run_name exp01
```

DDP 训练时 `gradient_accumulation_steps` 会自动按卡数等比缩减，保持全局等效 batch size 不变（32）。例如 4 卡时 accum=8，8 卡时 accum=4。

或使用训练脚本快速启动（更多参数修改直接编辑脚本）：

```bash
bash scripts/train_ddp.sh
```

训练过程中会在解析后的实验目录中保存：

```text
<run_dir>/latest_train.ckpt
<run_dir>/last.ckpt
<run_dir>/best_auc.ckpt
<run_dir>/metrics_train.csv
<run_dir>/metrics_val.csv
<run_dir>/tensorboard/
```

启动时终端会打印实际路径：`Output directory: <run_dir>`。

其中：

- `latest_train.ckpt`：当前 epoch 的 train 已完成、validation 未完成时的恢复点；resume 后只补 validation
- `last.ckpt`：train + validation + checkpoint 全部完成后的完整 epoch checkpoint
- `best_auc.ckpt`：按 FF++ validation ROC AUC 选择的最好 checkpoint
- `metrics_train.csv`：训练指标日志（loss、center distance 等）
- `metrics_val.csv`：验证指标日志（ROC AUC、PR AUC、accuracy 等）
- `tensorboard/`：TensorBoard 日志

查看 TensorBoard：

```bash

tensorboard --logdir <run_dir>/tensorboard
```

## 8. 断点恢复训练

推荐从完整 checkpoint 恢复：

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01 \
  --resume <run_dir>/last.ckpt
```

如果训练在 validation 阶段中断，`latest_train.ckpt` 会跳过已完成的 train，只补该 epoch 的 validation，保证 TensorBoard 的 validation 曲线不会缺点：

```bash
python train.py \
  --config configs/unite_ffpp_c23.yaml \
  --device cuda:0 \
  --run_name exp01 \
  --resume <run_dir>/latest_train.ckpt
```

使用脚本时，编辑 `scripts/train_ddp.sh` 的 `RUN_NAME` 和 `RESUME` 字段再运行。恢复同一个实验时，`RUN_NAME` 应与原训练保持一致。TensorBoard 默认写入同一个 `<run_dir>/tensorboard/` 目录，resume 后会读取 checkpoint 中的 `writer_step` 和 `global_step`：训练 loss 曲线按 `writer_step` 连续追加，学习率/梯度范数按 `global_step` 连续追加，validation 曲线按 epoch 补点。

只加载你自己训练生成的 checkpoint。项目为了兼容 PyTorch 2.6+ 的 `torch.load` 安全默认值，会用 `weights_only=False` 读取本地 checkpoint。

## 9. 评估

验证集评估：

```bash

python eval.py \
  --config configs/unite_ffpp_c23.yaml \
  --ckpt <run_dir>/best_auc.ckpt \
  --split val \
  --device cuda
```

测试集评估：

```bash
python eval.py \
  --config configs/unite_ffpp_c23.yaml \
  --ckpt <run_dir>/best_auc.ckpt \
  --split test \
  --device cuda
```

当前评估是 clip-level 二分类评估，输出指标包括：

- accuracy
- ROC AUC
- PR AUC / average precision
- precision@0.5
- recall@0.5
- precision@recall=0.8
- recall@precision=0.8

## 10. 跨域测试说明

论文的核心目标是通用检测，因此不要只看 FF++ 域内测试。FF++ only 训练完成后，可以继续准备外部数据集做跨域评估。

论文中适合跨域评估的数据集包括：

- Face manipulation：CelebDF、DeeperForensics、Deepfake-TIMIT、HifiFace、UADFV
- Background manipulation：AVID
- Fully synthetic：DeMamba；如果没有参与训练，GTA-V 也可作为 fully synthetic 测试
- In-the-wild：NYTimes DeepFake Quiz

当前仓库没有为这些外部数据集内置下载和格式转换脚本。后续接入时应保持与 FF++ split 类似的 CSV 字段，至少包含：

```text
abs_path,rel_path,label
```

然后复用 `eval.py` 做评估。跨域测试集不能混入 FF++ 训练集或验证集。

## 11. 训练创新版（GTA-free Universal Detector）

创新版在 baseline 基础上增加三个模块，旨在缓解 face-only 偏置并提升跨域泛化：

- 人脸掩蔽增强：用 OpenCV Haar cascade 在训练时构造 face-masked view
- Attention anti-collapse 损失：防止 attention 过度集中在少数帧
- Counterfactual consistency 损失：要求原图与 face-masked 图预测一致
- 类别平衡 CE：inverse-frequency 权重，缓解 fake-biased 预测

使用专门的 config：

```bash
# baseline（原版）
python train.py --config configs/unite_ffpp_c23.yaml --device cuda:0 \
  --run_name baseline_exp01

# 创新版（GTA-free）
python train.py --config configs/unite_ffpp_c23_gta_free.yaml --device cuda:0 \
  --run_name gta_free_exp01
```

DDP 多卡训练同理，在 `scripts/train_ddp.sh` 中修改 `CONFIG` 和 `RUN_NAME` 即可。也可以直接用创新版专属脚本：

```bash
bash scripts/train_gta_free.sh
```

创新版输出目录会有 `variant=gta_free` 层，与 baseline 的 `baseline` 层完全隔离。

## 12. 常见问题

### split 为空或提示缺失视频

通常是数据还没下载完成，或者 CSV 的 `File Path` 与实际数据根目录不匹配。确认视频是否位于：

```text
data/FaceForensics++_C23
```

并重新运行 split 构建命令。

### `ModuleNotFoundError: No module named 'torch'`

先安装依赖：

```bash

pip install -r requirements.txt
```

如果 PyTorch 版本与 CUDA 不匹配，请按你的 CUDA 版本重新安装 PyTorch。

### HuggingFace SigLIP 权重下载失败

默认模型是：

```text
google/siglip-so400m-patch14-384
```

如果训练机器不能联网，请提前下载并缓存该模型，或把 `configs/unite_ffpp_c23.yaml` 中的 `model.siglip_model_name` 改成本地路径。

### 显存不足

优先调整：

```yaml
train:
  batch_size: 1
  gradient_accumulation_steps: 32

model:
  encoder_batch_size: 1
```

SigLIP 对 64 帧视频片段的编码成本较高，单卡训练时建议从小步 smoke test 开始。

### DDP 报错 `Address already in use` 或端口冲突

`torchrun` 默认使用随机端口，如果冲突可以指定：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 train.py --config configs/unite_ffpp_c23.yaml
```

### 多卡训练时每卡显存占用比单卡高

DDP 会在 backward 时做 all-reduce 通信，额外消耗少量显存。如果 OOM，降低每卡 batch size 或保持 `batch_size: 1`。多卡训练本身已经通过减少 accumulation steps 来提速，不需要增大 batch size。

## 引用

如果你使用了本代码或认为它有帮助，请引用原始 UNITE 论文：

```bibtex
@inproceedings{kundu2025unite,
  author    = {Kundu, Rohit and Xiong, Hao and Mohanty, Vishal and Balachandran, Athula and Roy-Chowdhury, Amit K.},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title     = {Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content},
  year      = {2025},
  pages     = {28050-28060}
}
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
