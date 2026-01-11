"""
Stage 2: Test Basis Reconstruction Model
Evaluate trained video reconstruction model
"""

import os
import torch
import argparse
import numpy as np
import sys

from tqdm import tqdm

# Ensure the path includes the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.bases import AdaFNN
from model.bases_reconstruct import BasisReconstruction, BasisReconstruction_aug
from utils.path import pathManager
from utils.logger import setup_logger
from utils.dataloader import get_loader, DatasetConfig, LoaderConfig
from utils.metrics import HeartRateEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Test Bases-Guided rPPG Estimator (Stage 2)')
    
    # Test settings
    parser.add_argument('--test_protocol', type=str, nargs='+', default=['UBFC_test'],
                        help='Test protocol names')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the Stage 2 model weight file (.pth)')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for testing')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def load_model_for_test(checkpoint_path, device):
    """
    Load the full BasisReconstruction model from a Stage 2 checkpoint.
    This function automatically reconstructs the internal AdaFNN using saved args.
    """
    logger = setup_logger() # simple logger for inside function if needed
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Weight file not found: {checkpoint_path}")
        
    logger.info(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Retrieve arguments used during training
    saved_args = checkpoint['args']
    
    # Reconstruct time grid
    duration = saved_args.get('duration', 10.0)
    fps = saved_args.get('fps', 30)
    video_length = int(duration * fps)
    grid = np.linspace(0, duration, video_length)
    
    logger.info(f"Reconstructing model with duration={duration}s, fps={fps} (Length={video_length})")
    
    # 1. Reconstruct AdaFNN (Inner Model)
    # Note: parameters here are usually inherited from Stage 1 but saved in Stage 2 checkpoint args
    # if you propagated them. If Stage 2 args don't have n_base, we might need a fallback,
    # but assuming standard flow, they should be there or in a nested dict.
    
    # Fallback: check if args contains a nested 'basis_args' or similar. 
    # Based on stage_2_train logic, it seems we rely on args having these keys.
    # If keys are missing, defaults might be needed, but let's assume valid checkpoint.
    
    basis_model = AdaFNN(
        n_base=saved_args.get('n_base', 4),
        base_hidden=saved_args.get('base_hidden', [64, 64, 64]),
        grid=grid.tolist(),
        sub_hidden=saved_args.get('sub_hidden', [128, 128, 128]),
        dropout=saved_args.get('dropout', 0.1),
        lambda1=saved_args.get('lambda1', 0.0),
        lambda2=saved_args.get('lambda2', 0.0),
        device=device,
        n_frequencies=saved_args.get('n_frequencies', 32),
        freq_range=(saved_args.get('freq_min', 0.67), saved_args.get('freq_max', 3.0))
    ).to(device)
    
    # 2. Reconstruct BasisReconstruction (Outer Model)
    # Check if it was an augmented model
    do_augmentation = saved_args.get('do_augmentation', False)
    basis_scales = saved_args.get('basis_scales', [1.0])
    
    if do_augmentation:
        model = BasisReconstruction_aug(
            adafnn=basis_model,
            freeze_basis=True, # Always freeze for inference
            hidden_dim_estimator=128,
            scales=basis_scales
        ).to(device)
    else:
        model = BasisReconstruction(
            adafnn=basis_model,
            freeze_basis=True,
            hidden_dim_estimator=128
        ).to(device)
    
    # 3. Load State Dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, saved_args, video_length


def evaluate_model(model, dataloader, device, logger, hr_evaluator):
    """Evaluate model performance"""
    logger.info('\n' + '=' * 80)
    logger.info('Evaluating Model')
    logger.info('=' * 80)
    
    model.eval()
    all_refined_ppg = []
    all_gt_ppg = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for batch in pbar:
            video = batch['rgb'].to(device)
            gt_ppg = batch['gt'].to(device)
            
            # Forward pass
            final_rppg, bases_rppg, coarse_rppg = model(video)
            
            # Collect results (using bases_rppg as refined output)
            all_refined_ppg.append(bases_rppg.cpu()) 
            all_gt_ppg.append(gt_ppg.cpu())
    
    # Concatenate all data
    all_refined_ppg = torch.cat(all_refined_ppg, dim=0).numpy()
    all_gt_ppg = torch.cat(all_gt_ppg, dim=0).numpy()
    
    # Evaluate refined rPPG
    _, _, hr_metrics = hr_evaluator(all_refined_ppg, all_gt_ppg)
    
    # Calculate MSE
    mse = np.mean((all_refined_ppg - all_gt_ppg) ** 2)
    
    logger.info('\nTest Results:')
    logger.info('-' * 80)
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  MAE:  {hr_metrics['MAE']:.2f} BPM")
    logger.info(f"  RMSE: {hr_metrics['RMSE']:.2f} BPM")
    logger.info(f"  Pearson R: {hr_metrics['R']:.4f}")
    logger.info('=' * 80)
    
    return hr_metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Setup simple logger
    log_info_path, log_detail_path = pathManager.get_log_path(
        stage='test',
        model='BasisReconstruct',
        train_protocol=args.test_protocol
    )
    logger = setup_logger(
        info_path=str(log_info_path),
        detail_path=str(log_detail_path)
    )
    
    logger.info('=' * 80)
    logger.info('Starting Testing: Basis Reconstruction Model')
    logger.info('=' * 80)
    
    # Load Model
    try:
        model, saved_args, video_length = load_model_for_test(args.weights, device)
    except KeyError as e:
        logger.error(f"Error loading checkpoint: Missing key {e}. The checkpoint might be from an older version or Stage 1.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    logger.info(f"Model loaded. Video Length: {video_length}")

    # Prepare Test Dataset
    # We use arguments from the checkpoint to ensure data consistency (fps, duration)
    test_dataset_config = DatasetConfig(
        size=(128, 128),
        length=video_length,
        raw_length=int(video_length * 2), # Buffer for safety, though not needed for test
        sample=None,
        ratio=1.0,
        preload=False, # Usually False for testing to save RAM
        fixed_sample=True,
        augmentation={}
    )
    
    test_loader_config = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    test_loaders = get_loader(
        protocols=args.test_protocol,
        dataset_config=test_dataset_config,
        loader_config=test_loader_config,
        total_epochs=1
    )
    
    # HR Evaluator
    fps = saved_args.get('fps', 30)
    hr_evaluator = HeartRateEvaluator(Fs=fps, min_hr=40, max_hr=180)
    
    # Run Evaluation
    loader = test_loaders[0] if isinstance(test_loaders, list) else next(iter(test_loaders))
    evaluate_model(model, loader, device, logger, hr_evaluator)

if __name__ == '__main__':
    main()