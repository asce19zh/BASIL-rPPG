# BASIL-rPPG

**Basis Learning with Predictive rPPG Reconstruction for Heart Rate Estimation from Ultra-Short Facial Videos**

> Jhih-Wei Jhao, Wen-Pin Chen, Jun-Ren Chen, Yen-Chun Chou, Shih-Yu Yang, Pei-Kai Huang, Chiou-Ting Hsu
> ICPR 2026

BASIL-rPPG addresses heart rate (HR) estimation from 2-second facial video clips — a setting where existing methods fail due to insufficient periodic cues and temporal sparsity. The framework learns a compact set of physiological rPPG basis vectors (Stage 1) and then predicts mixing coefficients from facial videos to reconstruct the rPPG waveform (Stage 2).

---

## Method Overview

The framework follows a two-stage training paradigm:

**Stage 1 — rPPG Basis Learning**
A Basis Layer (AdaFNN) with learnable Fourier Feature Embeddings learns a compact set of rPPG basis vectors from ground-truth PPG signals. A Reconstruction Coefficient (RC) predictor is jointly trained to predict mixing coefficients from signals.

**Stage 2 — Basis-Driven Predictive Reconstruction**
With the basis frozen, a Feature Extractor (FE) and Coefficient Predictor (CP) are trained to map facial videos to reconstruction coefficients. An Auxiliary Estimator (AE) provides complementary unconstrained supervision. Dual-Level Temporal Augmentation (Basis Temporal Scaling + Video Temporal Resampling) expands heart rate coverage during training.

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (tested on NVIDIA RTX 4090)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Directory Structure

The project relies on an environment variable `DIR` pointing to a data root directory with the following layout:

```
$DIR/
├── hdf5/                    # HDF5-converted video data (auto-generated)
│   ├── UBFC-rPPG/
│   ├── PURE/
│   └── COHFACE/
├── protocol/                # Protocol files (JSON or TXT)
│   ├── UBFC_train.json
│   ├── UBFC_test.json
│   ├── PURE_train.json
│   ├── PURE_test.json
│   ├── COHFACE_train.json
│   └── COHFACE_test.json
├── weight/                  # Saved model checkpoints (auto-created)
│   ├── AdaFNN/
│   └── BasisReconstruct/
└── output/                  # Evaluation outputs (auto-created)
```

Create a `.env` file in the project root with:

```
DIR=/path/to/your/data/root
```

---

## Data Preparation

The dataloader expects data in HDF5 format under `$DIR/hdf5/`. If HDF5 files are not present, they are automatically created from cropped face images located at `$DIR/cropped/<DATASET>/RGB_crop/<video>/`.

Supported datasets:
- **UBFC-rPPG** — indoor, controlled lighting
- **PURE** — variable lighting and head motion
- **COHFACE** — compressed video, diverse subjects

### Protocol Files

Protocol files define which videos are used for training or testing. Place `.json` or `.txt` files in `$DIR/protocol/`. A `.txt` file uses comma-separated lines:

```
# dataset, video_name
UBFC-rPPG,subject1
UBFC-rPPG,subject2
```

For datasets with subfolders (e.g., COHFACE):
```
COHFACE,1,0
COHFACE,1,1
```

---

## Training

### Stage 1: rPPG Basis Learning

Train the AdaFNN model to learn physiological basis vectors from ground-truth PPG signals.

```bash
python stage_1_train.py \
  --train_protocol UBFC_train \
  --val_protocol UBFC_test \
  --duration 2.0 \
  --fps 30 \
  --n_base 4 \
  --n_frequencies 32 \
  --freq_min 0.67 \
  --freq_max 3.0 \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-3
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--train_protocol` | `UBFC_train` | Protocol name(s) for training |
| `--val_protocol` | `UBFC_test` | Protocol name(s) for validation |
| `--duration` | `2.0` | Clip duration in seconds |
| `--fps` | `30` | Video frame rate |
| `--n_base` | `4` | Number of basis vectors (K) |
| `--base_hidden` | `64 64 64` | Hidden sizes of basis micro-networks |
| `--sub_hidden` | `128 128 128` | Hidden sizes of RC predictor |
| `--n_frequencies` | `32` | Number of Fourier frequency components (M) |
| `--freq_min` | `0.67` | Min frequency in Hz (≈ 40 bpm) |
| `--freq_max` | `3.0` | Max frequency in Hz (≈ 180 bpm) |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--lambda1` | `0.01` | Sparsity regularization weight |
| `--lambda2` | `0.01` | Orthogonality regularization weight |
| `--resume` | `None` | Path to checkpoint to resume from |

The best checkpoint is saved to:
```
$DIR/weight/AdaFNN/<protocol>/length_060/epoch_9999.pth
```

---

### Stage 2: Basis-Driven Predictive Reconstruction

Train the video-to-coefficient mapping with the Stage 1 basis frozen.

```bash
python stage_2_train.py \
  --train_protocol UBFC_train \
  --val_protocol UBFC_test \
  --basis_checkpoint $DIR/weight/AdaFNN/ubfc_train/length_060/epoch_9999.pth \
  --duration 2.0 \
  --fps 30 \
  --epochs 300 \
  --batch_size 8 \
  --lr 1e-4
```

With Dual-Level Temporal Augmentation (DLTA) enabled:

```bash
python stage_2_train.py \
  --train_protocol UBFC_train \
  --val_protocol UBFC_test \
  --basis_checkpoint $DIR/weight/AdaFNN/ubfc_train/length_060/epoch_9999.pth \
  --do_augmentation \
  --basis_scales 0.5 0.75 1.0 1.33 2.0 \
  --compress_factor 1.0,2.8 \
  --epochs 300 \
  --batch_size 8 \
  --lr 1e-4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--basis_checkpoint` | required | Path to Stage 1 checkpoint |
| `--freeze_basis` | `True` | Freeze basis during Stage 2 training |
| `--do_augmentation` | `False` | Enable Dual-Level Temporal Augmentation |
| `--basis_scales` | `1.0` | Temporal scaling factors for Basis Temporal Scaling (BTS) |
| `--compress_factor` | `None` | Resampling factor range for Video Temporal Resampling (VTR), e.g. `1.0,2.8` |
| `--warmup` | `10` | Warmup epochs before augmentation activates |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--resume` | `None` | Path to checkpoint to resume from |

The best checkpoint is saved to:
```
$DIR/weight/BasisReconstruct/<protocol>/length_060/epoch_9999.pth
```

---

### Cross-Domain Training Example

To train on PURE + COHFACE and validate on UBFC-rPPG:

```bash
# Stage 1
python stage_1_train.py \
  --train_protocol PURE_train COHFACE_train \
  --val_protocol UBFC_test \
  --epochs 100

# Stage 2 (with DLTA)
python stage_2_train.py \
  --train_protocol PURE_train COHFACE_train \
  --val_protocol UBFC_test \
  --basis_checkpoint $DIR/weight/AdaFNN/pure_train,cohface_train/length_060/epoch_9999.pth \
  --do_augmentation \
  --basis_scales 0.5 0.75 1.0 1.33 2.0 \
  --compress_factor 1.0,2.8 \
  --epochs 300
```

---

## Evaluation

### Stage 1 Evaluation

```bash
python stage_1_test.py \
  --test_protocol UBFC_test \
  --weights $DIR/weight/AdaFNN/ubfc_train/length_060/epoch_9999.pth
```

### Stage 2 Evaluation

```bash
python stage_2_test.py \
  --test_protocol UBFC_test \
  --weights $DIR/weight/BasisReconstruct/ubfc_train/length_060/epoch_9999.pth
```

Evaluation metrics reported: **MAE** (bpm), **RMSE** (bpm), **Pearson R**.

---

## Results

Intra-domain performance on 2-second clips:

| Method | UBFC MAE | UBFC R | PURE MAE | PURE R | COHFACE MAE | COHFACE R |
|---|---|---|---|---|---|---|
| PhysFormer (CVPR'22) | 7.38 | 0.59 | 6.56 | 0.43 | 11.87 | 0.20 |
| RhythmMamba (AAAI'25) | 5.18 | 0.89 | 2.62 | 0.80 | 11.06 | 0.18 |
| **BASIL-rPPG (Ours)** | **4.47** | **0.91** | **2.56** | **0.91** | **6.86** | **0.79** |

---

## Logs

Training logs are saved under `./log/`:
```
./log/train/<protocol>/<model>/info_<timestamp>.log
./log/train/<protocol>/<model>/detail_<timestamp>.log
```

---

## Citation

```bibtex
@inproceedings{jhao2026basil,
  title     = {BASIL-rPPG: Basis Learning with Predictive rPPG Reconstruction for Heart Rate Estimation from Ultra-Short Facial Videos},
  author    = {Jhih-Wei Jhao and Wen-Pin Chen and Jun-Ren Chen and Yen-Chun Chou and Shih-Yu Yang and Pei-Kai Huang and Chiou-Ting Hsu},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```
