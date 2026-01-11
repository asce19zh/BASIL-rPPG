"""
Stage 1: Train AdaFNN Basis Functions
Train AdaFNN to learn basis functions for reconstructing PPG signals
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure these imports match your file structure (e.g., if you used the single models.py, change to: from models import AdaFNN)
from model.bases import AdaFNN 
from utils.path import pathManager
from utils.logger import setup_logger
from utils.dataloader import get_loader, DatasetConfig, LoaderConfig
from utils.metric import HeartRateEvaluator
from model.loss import NegativePearsonLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train AdaFNN Basis Functions (Stage 1)')
    
    # Dataset settings
    parser.add_argument('--train_protocol', type=str, nargs='+', default=['UBFC_train'],
                        help='Training protocol names')
    parser.add_argument('--val_protocol', type=str, nargs='+', default=['UBFC_test'],
                        help='Validation protocol names')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Duration of video clips in seconds')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second of the dataset')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Ratio of samples to use from each video')
    parser.add_argument('--sample_per_video', type=int, default=None,
                        help='Number of samples per video (None for auto)')
    parser.add_argument('--preload', action='store_true', default=False,
                        help='Preload dataset into memory')
    parser.add_argument('--fixed_sample', action='store_true', default=True,
                        help='Use fixed sampling strategy')
    
    # Model settings
    parser.add_argument('--n_base', type=int, default=4,
                        help='Number of basis functions')
    parser.add_argument('--base_hidden', type=int, nargs='+', default=[64, 64, 64],
                        help='Hidden layer sizes for basis networks')
    parser.add_argument('--sub_hidden', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes for coefficient prediction network')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--lambda1', type=float, default=0.01,
                        help='Sparsity regularization weight')
    parser.add_argument('--lambda2', type=float, default=0.01,
                        help='Orthogonality regularization weight')
    parser.add_argument('--n_frequencies', type=int, default=32,
                        help='Number of Fourier frequency components')
    parser.add_argument('--freq_min', type=float, default=0.67,
                        help='Minimum frequency in Hz (40 BPM)')
    parser.add_argument('--freq_max', type=float, default=3.0,
                        help='Maximum frequency in Hz (180 BPM)')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--l1_k', type=int, default=2,
                        help='Number of bases for L1 regularization')
    parser.add_argument('--l2_pairs', type=int, default=32,
                        help='Number of pairs for L2 regularization')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Pin memory for faster data transfer')
    
    # Checkpoint settings
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, args, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_r1_loss = 0
    total_r2_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    # pearson_loss = NegativePearsonLoss() # Unused in this snippet
    
    for batch in pbar:
        # Get ground truth PPG signal
        gt_ppg = batch['gt'].to(device)  # (B, J)
        
        # Forward pass
        pred_ppg, coefficient = model(gt_ppg)  # (B, J)
        
        # Reconstruction loss (MSE)
        rec_loss = torch.nn.functional.mse_loss(pred_ppg, gt_ppg)
        
        # Regularization loss
        r1_loss = model.R1(args.l1_k)  # Sparsity
        r2_loss = model.R2(args.l2_pairs)  # Orthogonality
        
        # Total loss
        loss = rec_loss + r1_loss + r2_loss 
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_r1_loss += r1_loss.item()
        total_r2_loss += r2_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{rec_loss.item():.4f}',
            'r1': f'{r1_loss.item():.4f}',
            'r2': f'{r2_loss.item():.4f}'
        })
    
    # Average loss
    avg_loss = total_loss / num_batches
    avg_rec_loss = total_rec_loss / num_batches
    avg_r1_loss = total_r1_loss / num_batches
    avg_r2_loss = total_r2_loss / num_batches
    
    logger.info(f'Train Loss: {avg_loss:.4f} (Rec: {avg_rec_loss:.4f}, R1: {avg_r1_loss:.4f}, R2: {avg_r2_loss:.4f})')
    logger.debug(f'Detailed - Batches: {num_batches}, Total Loss: {total_loss:.4f}')
    
    return avg_loss, avg_rec_loss, avg_r1_loss, avg_r2_loss


def validate(model, dataloader, device, logger, hr_evaluator=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_pred_ppg = []
    all_gt_ppg = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch in pbar:
            gt_ppg = batch['gt'].to(device)
            
            # Forward pass
            pred_ppg, coefficient = model(gt_ppg)
            
            # Reconstruction loss
            loss = torch.nn.functional.mse_loss(pred_ppg, gt_ppg)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for HR evaluation
            all_pred_ppg.append(pred_ppg.cpu())
            all_gt_ppg.append(gt_ppg.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    logger.info(f'Val Loss: {avg_loss:.4f}')
    
    # Calculate Heart Rate metrics (if evaluator is provided)
    hr_metrics = None
    if hr_evaluator is not None:
        all_pred_ppg = torch.cat(all_pred_ppg, dim=0).numpy()
        all_gt_ppg = torch.cat(all_gt_ppg, dim=0).numpy()
        _, _, hr_metrics = hr_evaluator(all_pred_ppg, all_gt_ppg)
        logger.info(f"HR Metrics - MAE: {hr_metrics['MAE']:.2f}, RMSE: {hr_metrics['RMSE']:.2f}, R: {hr_metrics['R']:.3f}")
    
    return avg_loss, hr_metrics


def save_checkpoint(model, optimizer, epoch, args, is_best=False):
    """Save checkpoint"""
    # Convert args to dict and add necessary parameters
    args_dict = vars(args).copy()
    args_dict['duration'] = args.duration
    args_dict['fps'] = args.fps
    args_dict['video_length'] = args.video_length
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args_dict
    }
    
    # Save latest checkpoint
    save_path = pathManager.get_weight_path(
        model='AdaFNN',
        protocol=args.train_protocol,
        length=args.video_length,
        epoch=epoch
    )
    torch.save(checkpoint, save_path)
    
    # If it is the best model, save additionally
    if is_best:
        best_path = pathManager.get_weight_path(
            model='AdaFNN',
            protocol=args.train_protocol,
            length=args.video_length,
            epoch=9999  # Use special epoch number for best model
        )
        torch.save(checkpoint, best_path)


def main():
    # Parse arguments
    args = parse_args()

    device = torch.device(args.device)

    args.video_length = int(args.duration * args.fps)
    args.raw_video_length = int(args.duration * args.fps)
    
    # Setup logger
    log_info_path, log_detail_path = pathManager.get_log_path(
        stage='train',
        model='AdaFNN',
        train_protocol=args.train_protocol
    )
    logger = setup_logger(
        info_path=str(log_info_path),
        detail_path=str(log_detail_path)
    )
    
    logger.info('=' * 80)
    logger.info('Starting Training: AdaFNN Basis Functions')
    logger.info('=' * 80)
    logger.info(f'Log files:')
    logger.info(f'  - Info: {log_info_path}')
    logger.info(f'  - Detail: {log_detail_path}')
    logger.info(f'Model weights will be saved to:')
    logger.info(f'  - {pathManager.dir}/weight/AdaFNN/{",".join(args.train_protocol)}/length_{args.video_length:03d}/')
    logger.info('=' * 80)
    
    grid = np.linspace(0, args.duration, args.video_length)
    
    logger.info(f'Video configuration: {args.duration}s @ {args.fps}fps = {args.video_length} frames')
    logger.info(f'Grid range: [{grid[0]:.4f}, {grid[-1]:.4f}] with {len(grid)} points')
    
    # Create model
    logger.info('Creating AdaFNN model...')
    model = AdaFNN(
        n_base=args.n_base,
        base_hidden=args.base_hidden,
        grid=grid,
        sub_hidden=args.sub_hidden,
        dropout=args.dropout,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=device,
        n_frequencies=args.n_frequencies,
        freq_range=(args.freq_min, args.freq_max)
    ).to(device)
    
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min
    )
    
    # Load checkpoint (if provided)
    start_epoch = 0
    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Prepare datasets
    logger.info('Loading datasets...')
    
    train_dataset_config = DatasetConfig(
        size=(128, 128),
        length=args.video_length,
        raw_length=args.raw_video_length,
        sample=args.sample_per_video,
        ratio=args.sample_ratio,
        preload=args.preload,
        fixed_sample=args.fixed_sample,
        augmentation={}
    )
    
    train_loader_config = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory
    )
    
    train_loaders = get_loader(
        protocols=args.train_protocol,
        dataset_config=train_dataset_config,
        loader_config=train_loader_config,
        total_epochs=args.epochs
    )
    
    # Validation dataset
    val_dataset_config = DatasetConfig(
        size=(128, 128),
        length=args.video_length,
        raw_length=args.raw_video_length,
        sample=None,
        ratio=1.0,
        preload=args.preload,
        fixed_sample=True,
        augmentation={}
    )
    
    val_loader_config = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory
    )
    
    val_loaders = get_loader(
        protocols=args.val_protocol,
        dataset_config=val_dataset_config,
        loader_config=val_loader_config,
        total_epochs=args.epochs
    )
    
    # Heart rate evaluator
    hr_evaluator = HeartRateEvaluator(Fs=30, min_hr=40, max_hr=180)
    
    # Training loop
    best_val_loss = float('inf')
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f'\nEpoch [{epoch+1}/{args.epochs}]')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Get dataloader for current epoch
        if isinstance(train_loaders, list):
            train_loader = train_loaders[epoch]
        else:
            train_loader = next(iter(train_loaders))
        
        if isinstance(val_loaders, list):
            val_loader = val_loaders[epoch]
        else:
            val_loader = next(iter(val_loaders))
        
        # Train
        train_loss, train_rec, train_r1, train_r2 = train_epoch(
            model, train_loader, optimizer, device, args, logger
        )
        
        # Validate
        val_loss, hr_metrics = validate(model, val_loader, device, logger, hr_evaluator)
        
        # Combined validation loss metric for checkpointing
        current_val_score = hr_metrics['MAE'] + hr_metrics['RMSE'] + (1 - hr_metrics['R'])
        
        # Update LR
        scheduler.step()
        
        # Save checkpoint
        is_best = current_val_score < best_val_loss
        if is_best:
            best_val_loss = current_val_score
            logger.info(f'New best validation score: {best_val_loss:.4f}')
            save_checkpoint(model, optimizer, epoch, args, is_best)
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, args, is_best)
            logger.info(f'Checkpoint saved at epoch {epoch+1}')
    
    logger.info('=' * 80)
    logger.info('Training completed!')
    logger.info(f'Best validation score: {best_val_loss:.4f}')
    logger.info('=' * 80)

if __name__ == '__main__':
    main()