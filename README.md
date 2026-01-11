# BASIL-rPPG: Basis Learning with Predictive rPPG Reconstruction for Heart Rate Estimation from Ultra-Short Facial Videos

The framework operates in two stages:
1.  **Stage 1:** Learning a set of periodic **Basis Functions** from ground truth PPG signals.
2.  **Stage 2:** Training a **Video Encoder** to predict coefficients, which are combined with the frozen basis functions to reconstruct the rPPG signal from facial video clips.

## Structure

```
├── dataloader.py       # VTR
├── models.py         # BASIL-rPPG model 
├── stage_1_train.py          # Training script for stage 1
├── stage_1_test.py           # Testing script for stage 1
├── stage_2_train.py          # Training script for stage 2
├── stage_2_test.py           # Testing script for stage 2
└── requirements.txt  # Dependencies
```

