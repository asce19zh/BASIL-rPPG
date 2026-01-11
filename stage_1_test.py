"""
Stage 1: Test AdaFNN Basis Functions
Evaluate trained AdaFNN model on test dataset
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from model.bases import AdaFNN
from utils.path import pathManager
from utils.logger import setup_logger
from utils.dataloader import get_loader, DatasetConfig, LoaderConfig
from utils.metric import HeartRateEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Test AdaFNN Basis Functions (Stage 1)')
    
    # Test settings
    parser.add_argument('--test_protocol', type=str, nargs='+', default=['UBFC_test'],
                        help='Test protocol names')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the model weight file (.pth)')
    
    # Data settings
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Duration of video clips in seconds')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second of the dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    
    # Model settings (Must match the trained model)
    parser.add_argument('--n_base', type=int, default=4,
                        help='Number of basis functions')
    parser.add_argument('--base_hidden', type=int, nargs='+', default=[64, 64, 64],
                        help='Hidden layer sizes for basis networks')
    parser.add_argument('--sub_hidden', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes for coefficient prediction network')
    parser.add_argument('--n_frequencies', type=int, default=32,
                        help='Number of Fourier frequency components')
    parser.add_argument('--freq_min', type=float, default=0.67,
                        help='Minimum frequency in Hz (40 BPM)')
    parser.add_argument('--freq_max', type=float, default=3.0,
                        help='Maximum frequency in Hz (180 BPM)')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def evaluate_best_model(model, dataloader, device, logger, hr_evaluator):
    """Evaluate model performance"""
    logger.info('\n' + '=' * 80)
    logger.info('Evaluating Model')
    logger.info('=' * 80)
    
    model.eval()
    all_pred_ppg = []
    all_gt_ppg = []
    total_mse = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Final Evaluation')
        for batch in pbar:
            gt_ppg = batch['gt'].to(device)
            
            # Forward pass
            pred_ppg, coefficient = model(gt_ppg)
            
            # Calculate MSE
            mse = torch.nn.functional.mse_loss(pred_ppg, gt_ppg)
            total_mse += mse.item()
            num_batches += 1
            
            # Collect predictions
            all_pred_ppg.append(pred_ppg.cpu())
            all_gt_ppg.append(gt_ppg.cpu())
    
    avg_mse = total_mse / num_batches
    
    # Concatenate all data
    all_pred_ppg = torch.cat(all_pred_ppg, dim=0).numpy()
    all_gt_ppg = torch.cat(all_gt_ppg, dim=0).numpy()
    
    # Evaluate heart rate
    _, _, hr_metrics = hr_evaluator(all_pred_ppg, all_gt_ppg)
    
    logger.info('\nEvaluation Results:')
    logger.info('-' * 80)
    logger.info(f"MSE:  {avg_mse:.6f}")
    logger.info(f"MAE:  {hr_metrics['MAE']:.2f} BPM")
    logger.info(f"RMSE: {hr_metrics['RMSE']:.2f} BPM")
    logger.info(f"Pearson R: {hr_metrics['R']:.4f}")
    logger.info('=' * 80)
    
    return hr_metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Calculate video length
    args.video_length = int(args.duration * args.fps)
    args.raw_video_length = int(args.duration * args.fps)
    
    # Setup simple logger
    log_info_path, log_detail_path = pathManager.get_log_path(
        stage='test',
        model='AdaFNN',
        train_protocol=args.test_protocol
    )
    logger = setup_logger(
        info_path=str(log_info_path),
        detail_path=str(log_detail_path)
    )
    
    logger.info('=' * 80)
    logger.info('Starting Testing: AdaFNN Basis Functions')
    logger.info(f'Loading weights from: {args.weights}')
    logger.info('=' * 80)
    
    # Time grid setup
    grid = np.linspace(0, args.duration, args.video_length)
    
    # Create model
    logger.info('Creating AdaFNN model...')
    model = AdaFNN(
        n_base=args.n_base,
        base_hidden=args.base_hidden,
        grid=grid,
        sub_hidden=args.sub_hidden,
        dropout=0.0, # No dropout during inference
        lambda1=0.0,
        lambda2=0.0,
        device=device,
        n_frequencies=args.n_frequencies,
        freq_range=(args.freq_min, args.freq_max)
    ).to(device)
    
    # Load weights
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        logger.error(f"No weight file found at {args.weights}")
        return

    # Prepare Test Data
    test_dataset_config = DatasetConfig(
        size=(128, 128),
        length=args.video_length,
        raw_length=args.raw_video_length,
        sample=None,
        ratio=1.0,
        preload=False,
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
        total_epochs=1 # Only need one epoch for testing
    )
    
    # Heart rate evaluator
    hr_evaluator = HeartRateEvaluator(Fs=30, min_hr=40, max_hr=180)
    
    # Run evaluation
    loader = test_loaders[0] if isinstance(test_loaders, list) else next(iter(test_loaders))
    evaluate_best_model(model, loader, device, logger, hr_evaluator)

if __name__ == '__main__':
    main()