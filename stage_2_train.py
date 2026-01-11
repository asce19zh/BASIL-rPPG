"""
Stage 2: Train Basis Reconstruction Model
Train video reconstruction model using pre-trained AdaFNN basis functions
Encoder -> Coefficient Estimator -> Basis Reconstruction
"""

import os
import torch
import argparse
import numpy as np
import math
import torch.nn.functional as F # Added missing import for interpolate/pad
import sys

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure the path includes the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.bases import AdaFNN
from model.bases_reconstruct import BasisReconstruction, BasisReconstruction_aug
from utils.path import pathManager
from utils.logger import setup_logger
from utils.dataloader import get_loader, DatasetConfig, LoaderConfig
from utils.metrics import HeartRateEvaluator
from utils.metric import NegativePearsonLoss
from model.loss import CrossEntropyPSDLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bases-Guided rPPG Estimator (Stage 2)')
    
    # Dataset settings
    parser.add_argument('--train_protocol', type=str, nargs='+', default=['UBFC_train'],
                        help='Training protocol names')
    parser.add_argument('--val_protocol', type=str, nargs='+', default=['UBFC_test'],
                        help='Validation protocol names')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Duration of video clips in seconds')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second of the video')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Ratio of samples to use from each video')
    parser.add_argument('--sample_per_video', type=int, default=None,
                        help='Number of samples per video (None for auto)')
    parser.add_argument('--preload', action='store_true', default=True,
                        help='Preload dataset into memory')
    parser.add_argument('--fixed_sample', action='store_true', default=True,
                        help='Use fixed sampling strategy')
    
    # Basis model settings (Load from Stage 1)
    parser.add_argument('--basis_checkpoint', type=str, required=True,
                        help='Path to trained AdaFNN checkpoint')
    parser.add_argument('--freeze_basis', action='store_true', default=True,
                        help='Freeze basis functions during training')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
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
    parser.add_argument('--warmup', type=int, default=10,
                        help='warmup epochs')
    parser.add_argument('--do_augmentation', action='store_true', default=False,
                        help='augment basis')
    parser.add_argument(
        '--compress_factor',
        type=str,                 # Changed to str to accept ranges like "1.0,2.8"
        default=None,
        help='Compression factor for temporal HR augmentation. Example: 2.0 or 1.0,2.8'
    )
    parser.add_argument(
        '--basis_scales',
        type=float,
        nargs='+',
        default=[1.0],
        help='temporal scaling factors for basis augmentation'
    )

    return parser.parse_args()

def parse_compress_factor(cf):
    if cf is None:
        return None

    # case 1: Single value, e.g., "2.0"
    if ',' not in cf:
        return float(cf)

    # case 2: Range, e.g., "1.5,2.8"
    parts = cf.split(',')
    if len(parts) != 2:
        raise ValueError("compress_factor must be float or 'min,max'")
    
    return [float(parts[0]), float(parts[1])]

def load_basis_model(checkpoint_path, device):
    """Load AdaFNN model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get AdaFNN config from checkpoint
    saved_args = checkpoint['args']
    
    # Reconstruct time grid (using duration and fps)
    duration = saved_args.get('duration', 10.0)
    fps = saved_args.get('fps', 30)
    video_length = int(duration * fps)
    raw_video_length = int(duration * fps * 2) # data augmentation buffer
    grid = np.linspace(0, duration, video_length)
    
    # Create AdaFNN model
    basis_model = AdaFNN(
        n_base=saved_args['n_base'],
        base_hidden=saved_args['base_hidden'],
        grid=grid.tolist(),
        sub_hidden=saved_args['sub_hidden'],
        dropout=saved_args['dropout'],
        lambda1=saved_args['lambda1'],
        lambda2=saved_args['lambda2'],
        device=device,
        n_frequencies=saved_args['n_frequencies'],
        freq_range=(saved_args['freq_min'], saved_args['freq_max'])
    ).to(device)
    
    # Load weights
    basis_model.load_state_dict(checkpoint['model_state_dict'])
    
    return basis_model, saved_args, video_length, raw_video_length

def aug_prob(epoch, total_epochs):
    if epoch < 0.2 * total_epochs:
        return 0.0
    elif epoch < 0.5 * total_epochs:
        return 0.3
    else:
        return 0.5

def train_epoch(model, dataloader, optimizer, device, args, logger, loss_fn, loss_psd, epoch_now , warmup_epochs):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_rec_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Get input and ground truth
        video = batch['rgb'].to(device)  # (B, C, T, H, W)
        gt_ppg = batch['gt'].to(device)  # (B, J)

        p_aug = aug_prob(epoch_now, args.epochs)
        use_aug = (
            args.compress_factor is not None
            and torch.rand(1).item() < p_aug
        )
        if use_aug:
            video = batch['rgb_aug'].to(device)
            gt_ppg = batch['gt_aug'].to(device)

        # === 1. forward on anchor ===
        final_rppg, rppg, coarse_rppg = model(video, epoch_now, args.epochs)

        loss_coarse_p = loss_fn(coarse_rppg, gt_ppg) # abla coarse_rppg
        loss_align  = loss_fn(rppg, coarse_rppg.detach())

        # === 3. Combined loss ===
        loss = loss_coarse_p + loss_align
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'loss_align': f'{loss_align.item():.4f}',
            'coarse':f'{loss_coarse_p.item():.4f}'

        })
    
    # Average loss
    avg_loss = total_loss / num_batches
    
    logger.info(f'Train Loss: {avg_loss:.4f}')
    logger.debug(f'Detailed - Batches: {num_batches}, Total Loss: {total_loss:.4f}')
    
    return avg_loss


def validate(model, dataloader, device, args, logger, loss_fn, epoch_now, warmup_epochs, hr_evaluator=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_pred_ppg = []
    all_gt_ppg = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        loss_psd = CrossEntropyPSDLoss(fs=args.fps)
        for batch in pbar:
            video = batch['rgb'].to(device)
            gt_ppg = batch['gt'].to(device)
            
            # Forward pass
            final_rppg, rppg, coarse_rppg = model(video)
   
            # Calculate Loss
            # loss_rec = loss_psd(rppg, gt_ppg)
            loss_coarse_p = loss_fn(coarse_rppg, gt_ppg) # abla coarse_rppg
            # loss_rec_p = loss_fn(rppg, gt_ppg)
            loss_align = loss_fn(rppg, coarse_rppg.detach())

            # abla 
            loss =  loss_coarse_p + loss_align 
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            all_pred_ppg.append(rppg.cpu()) # abla
            all_gt_ppg.append(gt_ppg.cpu())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
    
    avg_loss = total_loss / num_batches
    
    logger.info(f'Val Loss: {avg_loss:.4f}')
    
    # Calculate HR metrics
    hr_metrics = None
    if hr_evaluator is not None:
        pred_signals = torch.cat(all_pred_ppg, dim=0).numpy()
        gt_signals = torch.cat(all_gt_ppg, dim=0).numpy()
        
        # Evaluate rPPG
        pred_hr, gt_hr, results = hr_evaluator(pred_signals, gt_signals)
        logger.info(f' MAE: {results["MAE"]:-3.4f}, RMSE: {results["RMSE"]:-3.4f}, R: {results["R"]:-3.4f}')
    return avg_loss, results


def evaluate_best_model(model, dataloader, device, logger, hr_evaluator):
    """Evaluate full performance of the best model"""
    logger.info('\n' + '=' * 80)
    logger.info('Evaluating Best Model')
    logger.info('=' * 80)
    
    model.eval()
    all_refined_ppg = []
    all_gt_ppg = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Final Evaluation')
        for batch in pbar:
            video = batch['rgb'].to(device)
            gt_ppg = batch['gt'].to(device)
            
            # Forward pass
            final_rppg, bases_rppg, coarse_rppg = model(video)
            
            # Collect results
            all_refined_ppg.append(bases_rppg.cpu()) # abla
            all_gt_ppg.append(gt_ppg.cpu())
    
    # Concatenate all data
    all_refined_ppg = torch.cat(all_refined_ppg, dim=0).numpy()
    all_gt_ppg = torch.cat(all_gt_ppg, dim=0).numpy()
    
    # Evaluate refined rPPG
    _, _, hr_metrics = hr_evaluator(all_refined_ppg, all_gt_ppg)
    
    # Calculate MSE
    mse = np.mean((all_refined_ppg - all_gt_ppg) ** 2)
    
    logger.info('\nFinal Evaluation Results:')
    logger.info('-' * 80)
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  MAE:  {hr_metrics['MAE']:.2f} BPM")
    logger.info(f"  RMSE: {hr_metrics['RMSE']:.2f} BPM")
    logger.info(f"  Pearson R: {hr_metrics['R']:.4f}")
    logger.info('=' * 80)
    
    return hr_metrics


def save_checkpoint(model, optimizer, epoch, args, is_best=False):
    args_dict = vars(args).copy()
    args_dict['duration'] = args.duration
    args_dict['fps'] = args.fps
    args_dict['video_length'] = args.video_length
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args_dict
    }
    
    # Save latest checkpoint
    save_path = pathManager.get_weight_path(
        model='BasisReconstruct',
        protocol=args.train_protocol,
        length=args.video_length,
        epoch=epoch
    )
    torch.save(checkpoint, save_path)
    
    # If best model, save additionally
    if is_best:
        best_path = pathManager.get_weight_path(
            model='BasisReconstruct',
            protocol=args.train_protocol,
            length=args.video_length,
            epoch=9999
        )
        torch.save(checkpoint, best_path)


def main():
    # Parse arguments
    args = parse_args()
    
    args.video_length = int(args.duration * args.fps)
    args.raw_video_length = int(args.duration * args.fps) * 2
    
    # Setup logger
    log_info_path, log_detail_path = pathManager.get_log_path(
        stage='train',
        model='BasisReconstruct',
        train_protocol=args.train_protocol
    )
    logger = setup_logger(
        info_path=str(log_info_path),
        detail_path=str(log_detail_path)
    )
    
    logger.info('=' * 80)
    logger.info('Starting Training: Basis Reconstruction Model')
    logger.info('=' * 80)
    logger.info(f'Log files:')
    logger.info(f'  - Info: {log_info_path}')
    logger.info(f'  - Detail: {log_detail_path}')
    logger.info(f'Model weights will be saved to:')
    logger.info(f'  - weight/BasisReconstruct/{",".join(args.train_protocol)}/length_{args.video_length:03d}/')
    logger.info('=' * 80)

    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # Calculate video_length
    logger.info(f'Video configuration: {args.duration}s @ {args.fps}fps = {args.video_length} frames')
    
    # Load basis model
    logger.info(f'Loading basis model from: {args.basis_checkpoint}')
    basis_model, basis_args, basis_video_length, raw_video_length = load_basis_model(args.basis_checkpoint, device)
    logger.info(f'Basis model loaded successfully (n_base={basis_args["n_base"]}, length={basis_video_length})')
    
    # Check video length consistency
    if args.video_length != basis_video_length:
        logger.warning(f'Warning: video_length mismatch! Training: {args.video_length}, Basis: {basis_video_length}')
        logger.warning(f'Using basis model length: {basis_video_length}')
        args.video_length = basis_video_length
        args.raw_video_length = raw_video_length
    
    # Create BasisReconstruction Model
    logger.info('Creating BasisReconstruction model...')
    if args.do_augmentation:
        logger.info('using bases augmentation...')
        model = BasisReconstruction_aug(
            adafnn=basis_model,
            freeze_basis=args.freeze_basis,
            hidden_dim_estimator=128,
            scales=args.basis_scales
        ).to(device)
    else:
        model = BasisReconstruction(
            adafnn=basis_model,
            freeze_basis=args.freeze_basis,
            hidden_dim_estimator=128
        ).to(device)
    
    if args.freeze_basis:
        logger.info('Basis functions are frozen')
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trainable parameters: {trainable_params:,} / {total_params:,}')
    
    # Create Optimizer (only for trainable parameters)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create LR Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min
    )
    
    # Load Checkpoint (if provided)
    start_epoch = 0
    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Prepare Datasets
    logger.info('Loading datasets...')
    compress_factor = parse_compress_factor(args.compress_factor)
    
    train_dataset_config = DatasetConfig(
        size=(128, 128),
        length=args.video_length,
        raw_length=args.raw_video_length,
        sample=args.sample_per_video,
        ratio=args.sample_ratio,
        preload=args.preload,
        fixed_sample=args.fixed_sample,
        augmentation={},
        compress_factor=compress_factor    
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
    
    # Validation Dataset
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

    loss_fn = NegativePearsonLoss().to(device)
    logger.info('Using Negative Pearson Loss')
    loss_psd = CrossEntropyPSDLoss(fs=args.fps).to(device)
    logger.info('Using Cross Entropy PSD Loss')
    # HR Evaluator
    hr_evaluator = HeartRateEvaluator(Fs=args.fps, min_hr=40, max_hr=180)
    
    # Training Loop
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
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args, logger, loss_fn, loss_psd, epoch, args.warmup
        )
        
        # Validate
        val_loss, hr_metrics = validate(
            model, val_loader, device, args, logger, loss_fn, epoch, args.warmup, hr_evaluator
        )
        
        val_loss = hr_metrics['MAE'] + hr_metrics['RMSE'] + (1 - hr_metrics['R'])*30   # Combined metric
        
        # Update LR
        scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f'New best validation loss: {best_val_loss:.4f}')
            save_checkpoint(model, optimizer, epoch, args, is_best)
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, args, is_best)
            logger.info(f'Checkpoint saved at epoch {epoch+1}')
    
    logger.info('=' * 80)
    logger.info('Training completed!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    logger.info('=' * 80)
    
    # Evaluate best model
    logger.info('\nEvaluating best model on validation set...')
    
    # Reload best model
    best_checkpoint_path = pathManager.get_weight_path(
        model='BasisReconstruct',
        protocol=args.train_protocol,
        length=args.video_length,
        epoch=9999
    )
    
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded best model from: {best_checkpoint_path}')
    else:
        logger.info('Using current best model state')
    
    # Evaluate on validation set
    val_loader = val_loaders[-1] if isinstance(val_loaders, list) else next(iter(val_loaders))
    hr_metrics_final = evaluate_best_model(
        model, val_loader, device, logger, hr_evaluator
    )


if __name__ == '__main__':
    main()