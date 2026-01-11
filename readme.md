# AdaFNN-rPPG: Periodic Basis Learning for Remote Heart Rate Estimation

This repository contains a PyTorch implementation for remote Photoplethysmography (rPPG) signal reconstruction using **Adaptive Fourier Neural Networks (AdaFNN)**.

The framework operates in two stages:
1.  **Stage 1:** Learning a set of periodic **Basis Functions** from ground truth PPG signals using AdaFNN.
2.  **Stage 2:** Training a **Video Encoder** to predict coefficients, which are combined with the frozen basis functions to reconstruct the rPPG signal from facial video clips.

## 📂 Project Structure

```text
.
├── models.py              # Neural network architectures (AdaFNN, Encoder, Estimator)
├── stage_1_train.py       # Training script for Stage 1 (Basis Learning)
├── stage_1_test.py        # Evaluation script for Stage 1
├── stage_2_train.py       # Training script for Stage 2 (Video Reconstruction)
├── stage_2_test.py        # Evaluation script for Stage 2
├── requirements.txt       # Python dependencies
└── utils/                 # Utilities for data loading, logging, and metrics
    ├── dataloader.py
    ├── logger.py
    ├── metric.py
    └── path.py
🛠️ InstallationClone the repository:Bashgit clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
Install dependencies (Python 3.8+ recommended):Bashpip install -r requirements.txt
🚀 Usage1. Data PreparationEnsure your dataset paths (e.g., UBFC-rPPG) are correctly configured in utils/path.py (or whichever file handles your path management). The dataloaders expect protocols named UBFC_train and UBFC_test by default.2. Stage 1: Basis LearningIn this stage, the model learns general periodic basis functions from the ground truth PPG signals.Training:Bashpython stage_1_train.py \
  --train_protocol UBFC_train \
  --val_protocol UBFC_test \
  --n_base 4 \
  --epochs 100 \
  --batch_size 16 \
  --device cuda
Testing:Bashpython stage_1_test.py \
  --test_protocol UBFC_test \
  --weights ./path/to/stage1_checkpoint.pth
3. Stage 2: rPPG ReconstructionIn this stage, the learned basis functions are frozen. The video encoder is trained to predict the optimal coefficients to reconstruct the signal.Training:Note: You must provide the checkpoint from Stage 1.Bashpython stage_2_train.py \
  --train_protocol UBFC_train \
  --val_protocol UBFC_test \
  --basis_checkpoint ./path/to/stage1_checkpoint.pth \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --do_augmentation \
  --compress_factor 1.0,2.0
Testing:Bashpython stage_2_test.py \
  --test_protocol UBFC_test \
  --weights ./path/to/stage2_checkpoint.pth
⚙️ Key ArgumentsArgumentDescriptionDefault--durationLength of video clips in seconds2.0--fpsFrames per second of the input video30--n_baseNumber of basis functions in AdaFNN4--n_frequenciesNumber of Fourier frequency components32--compress_factor(Stage 2) Temporal augmentation factor (float or range like 1.0,2.8)None--basis_scales(Stage 2) Scales for basis augmentation[1.0]--freeze_basis(Stage 2) Whether to freeze Stage 1 basis weightsTrue📊 PerformanceThe model evaluates performance using standard rPPG metrics:MAE (Mean Absolute Error)RMSE (Root Mean Square Error)r (Pearson Correlation Coefficient)📜 LicenseThis project is licensed under the MIT License.