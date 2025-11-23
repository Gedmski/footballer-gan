"""
Train a DCGAN + InfoGAN on footballer face dataset.
Uses DiffAugment, SpectralNorm, AMP, EMA, and TTUR optimizers.
"""
import os
import argparse
from pathlib import Path
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision.utils import save_image

# Import project modules
from datasets.fm_cutout import get_fm_dataloader
from models.generator import build_generator, EMAWrapper
from models.discriminator import build_discriminator
from models.q_head import build_q_head
from losses.gan_losses import build_gan_loss, r1_regularization
from losses.infogan import build_infogan_loss, sample_categorical, sample_continuous
from augment.diffaugment import DiffAugment


def fix_nan_parameters(model, replacement_value=None):
    """Replace NaN and Inf values in model parameters with a safe replacement value."""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nan_mask = torch.isnan(param.data) | torch.isinf(param.data)
                if nan_mask.any():
                    if replacement_value is None:
                        # Use parameter mean or small random value as replacement
                        finite_values = param.data[~nan_mask]
                        if len(finite_values) > 0:
                            replacement = finite_values.mean().item()
                        else:
                            replacement = 1e-6  # Small positive value
                    else:
                        replacement = replacement_value

                    param.data[nan_mask] = replacement
                    print(f"Fixed {nan_mask.sum().item()} NaN/Inf values in model parameters (replaced with {replacement})")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_directories(config):
    """Create output directories."""
    paths = config['paths']
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(step, generator, discriminator, q_head, g_optimizer, d_optimizer,
                    ema, config, filename):
    """Save training checkpoint."""
    ckpt_path = Path(config['paths']['ckpts']) / filename
    
    checkpoint = {
        'step': step,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'q_head': q_head.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'ema': ema.state_dict() if ema else None,
        'config': config,
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


def load_checkpoint(path, generator, discriminator, q_head, g_optimizer,
                    d_optimizer, ema=None, device='cuda', config=None):
    """Load training checkpoint with transfer learning support."""
    checkpoint = torch.load(path, map_location=device)
    
    # Check if this is a transfer learning scenario (different image size)
    checkpoint_config = checkpoint.get('config')
    if checkpoint_config and config:
        checkpoint_img_size = checkpoint_config['model']['image_size']
        current_img_size = config['model']['image_size']
        
        if checkpoint_img_size != current_img_size:
            print(f"Transfer learning: {checkpoint_img_size}x{checkpoint_img_size} -> {current_img_size}x{current_img_size}")
            
            # Transfer learning for generator
            gen_state_dict = checkpoint['generator']
            current_gen_state_dict = generator.state_dict()
            
            # Copy matching layers (lower resolution layers)
            transferred_gen = {}
            for name, param in gen_state_dict.items():
                if name in current_gen_state_dict:
                    if param.shape == current_gen_state_dict[name].shape:
                        transferred_gen[name] = param
                        print(f"  G: Transferred {name}")
                    else:
                        print(f"  G: Skipped {name} (shape mismatch: {param.shape} vs {current_gen_state_dict[name].shape})")
                else:
                    print(f"  G: Skipped {name} (not in current model)")
            
            # Load transferred weights
            missing_keys, unexpected_keys = generator.load_state_dict(transferred_gen, strict=False)
            if missing_keys:
                print(f"  G: Missing keys (new layers): {missing_keys}")
            if unexpected_keys:
                print(f"  G: Unexpected keys: {unexpected_keys}")
            
            # Transfer learning for discriminator
            disc_state_dict = checkpoint['discriminator']
            current_disc_state_dict = discriminator.state_dict()
            
            transferred_disc = {}
            for name, param in disc_state_dict.items():
                if name in current_disc_state_dict:
                    if param.shape == current_disc_state_dict[name].shape:
                        transferred_disc[name] = param
                        print(f"  D: Transferred {name}")
                    else:
                        print(f"  D: Skipped {name} (shape mismatch: {param.shape} vs {current_disc_state_dict[name].shape})")
                else:
                    print(f"  D: Skipped {name} (not in current model)")
            
            # Load transferred weights
            missing_keys, unexpected_keys = discriminator.load_state_dict(transferred_disc, strict=False)
            if missing_keys:
                print(f"  D: Missing keys (new layers): {missing_keys}")
            if unexpected_keys:
                print(f"  D: Unexpected keys: {unexpected_keys}")
            
            # Q-head should be compatible (same feature_dim)
            q_head.load_state_dict(checkpoint['q_head'])
            print("  Q: Loaded Q-head weights")
            
            # Skip optimizer loading for transfer learning (different architectures)
            print("  Skipping optimizer loading for transfer learning")
            
            # Handle EMA if present
            if ema and checkpoint['ema']:
                # EMA also needs transfer learning
                ema_state_dict = checkpoint['ema']
                current_ema_state_dict = ema.shadow
                
                transferred_ema = {}
                for name, param in ema_state_dict.items():
                    if name in current_ema_state_dict:
                        if param.shape == current_ema_state_dict[name].shape:
                            transferred_ema[name] = param
                        else:
                            print(f"  EMA: Skipped {name} (shape mismatch)")
                    else:
                        print(f"  EMA: Skipped {name} (not in current model)")
                
                # Update EMA shadow with transferred weights
                ema.shadow = transferred_ema
                
                # Initialize any missing keys in EMA shadow with current model weights
                for name, param in generator.named_parameters():
                    if param.requires_grad and name not in ema.shadow:
                        ema.shadow[name] = param.data.clone()
                        print(f"  EMA: Initialized {name} with current weights")
                
                print("  EMA: Transferred EMA weights")
        else:
            # Normal loading (same architecture)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            q_head.load_state_dict(checkpoint['q_head'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            
            if ema and checkpoint['ema']:
                ema.load_state_dict(checkpoint['ema'])
    else:
        # Fallback for old checkpoints without config
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        q_head.load_state_dict(checkpoint['q_head'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        
        if ema and checkpoint['ema']:
            ema.load_state_dict(checkpoint['ema'])
    
    step = checkpoint['step']
    print(f"Checkpoint loaded from {path}, resuming at step {step}")
    return step


def generate_samples(generator, ema, config, num_samples=16, device='cuda', truncation_psi=0.8):
    """Generate sample images using EMA with truncation."""
    # Use EMA if available
    use_ema = ema is not None
    if use_ema:
        ema.apply_shadow()
    
    generator.eval()
    with torch.no_grad():
        # Truncated normal sampling
        z = torch.randn(num_samples, config['model']['z_dim'], device=device) * truncation_psi
        c_cat = sample_categorical(num_samples, config['model']['c_cat_dim'], device)
        c_cont = sample_continuous(num_samples, config['model']['c_cont_dim'], device=device)
        
        fake_images = generator(z, c_cat, c_cont)
        
        # Check for NaN/Inf in generated images
        if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
            print(f"Warning: NaN/Inf detected in generated samples, skipping sample save")
            # Return None to indicate failure
            if use_ema:
                ema.restore()
            return None
    
    generator.train()
    
    if use_ema:
        ema.restore()
    
    return fake_images


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in the directory."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    
    # Look for checkpoint_latest.pt first
    latest_ckpt = ckpt_dir / 'checkpoint_latest.pt'
    if latest_ckpt.exists():
        return str(latest_ckpt)
    
    # Look for numbered checkpoints
    ckpt_files = list(ckpt_dir.glob('checkpoint_step_*.pt'))
    if not ckpt_files:
        return None
    
    # Find the one with the highest step number
    latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split('_')[-1]))
    return str(latest_ckpt)


def train():
    """Main training loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Footballer FaceGAN')
    parser.add_argument('--config', type=str, default='configs/dcgan_infogan_128.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume-latest', action='store_true',
                        help='Automatically resume from the latest checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create directories
    create_directories(config)
    
    # Enable cudnn benchmark
    if config.get('cudnn_benchmark', True) and device == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Build dataloaders
    print("Loading dataset...")
    train_loader = get_fm_dataloader(config, is_train=True)
    print(f"Dataset size: {len(train_loader.dataset)} images")
    
    # Build models
    print("Building models...")
    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    q_head = build_q_head(config, discriminator.feature_dim).to(device)
    
    # Build EMA wrapper for generator
    ema = None
    if config['model']['g'].get('use_ema', True):
        ema = EMAWrapper(generator, decay=config['model']['g'].get('ema_decay', 0.999))
    
    # Build losses
    gan_loss_fn = build_gan_loss(config)
    infogan_loss_fn = build_infogan_loss(config)
    
    # Build optimizers
    g_opt_cfg = config['optim']['g']
    d_opt_cfg = config['optim']['d']
    
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=g_opt_cfg['lr'],
        betas=tuple(g_opt_cfg['betas']),
        weight_decay=g_opt_cfg.get('weight_decay', 0.0)
    )
    
    d_optimizer = torch.optim.Adam(
        list(discriminator.parameters()) + list(q_head.parameters()),
        lr=d_opt_cfg['lr'],
        betas=tuple(d_opt_cfg['betas']),
        weight_decay=d_opt_cfg.get('weight_decay', 0.0)
    )
    
    # AMP scaler - use safer settings for 256x256 stability
    if config.get('amp', True) and device == 'cuda':
        scaler = GradScaler('cuda', init_scale=2**10, growth_interval=2000)
    else:
        scaler = None
    
    # Get train config early
    train_cfg = config['train']
    
    # --- Stabilizers ---
    # Note: Learning rates now set from config, not hardcoded
    EMA_DECAY = 0.9995
    R1_GAMMA = 10.0
    G_UPS = 2
    REAL_LABEL = 0.9
    CLIP_NORM = 5.0
    # Auto recovery -> normal training switch (change threshold as needed)
    RECOVERY_END = train_cfg.get('recovery_end', 15000) if 'train' in locals() else 15000
    
    # Transfer learning stabilizers - extend recovery for new layers
    is_transfer_learning = config['train'].get('start_from') is not None
    if is_transfer_learning:
        RECOVERY_END = max(RECOVERY_END, 120000)  # Extend recovery to step 120k for transfer learning
        print(f"Transfer learning detected - extending recovery period to step {RECOVERY_END}")
    
    # Read update ratios from config
    g_updates_per_step = train_cfg.get('g_updates_per_step', 1)
    d_updates_per_step = train_cfg.get('d_updates_per_step', 1)
    
    def instance_noise_sigma(step, warmdown=20000, start=0.08):
        return max(start * (1.0 - step / float(warmdown)), 0.0)
    
    # Resume from checkpoint if specified
    start_step = 0
    resume_path = None
    
    if args.resume_latest:
        resume_path = find_latest_checkpoint(config['paths']['ckpts'])
        if resume_path:
            print(f"Auto-resuming from latest checkpoint: {resume_path}")
        else:
            print("No checkpoint found for auto-resume, starting from scratch")
    elif args.resume:
        resume_path = args.resume
    elif config['train'].get('start_from'):
        resume_path = config['train']['start_from']
        print(f"Resuming from config-specified checkpoint: {resume_path}")
    
    if resume_path:
        start_step = load_checkpoint(
            resume_path, generator, discriminator, q_head,
            g_optimizer, d_optimizer, ema, device, config
        )
        
        # Reset learning rates to config values after loading checkpoint
        # (checkpoint loading overwrites optimizer param_groups)
        g_opt_cfg = config['optim']['g']
        d_opt_cfg = config['optim']['d']
        for g in g_optimizer.param_groups:
            g['lr'] = g_opt_cfg['lr']
        for d in d_optimizer.param_groups:
            d['lr'] = d_opt_cfg['lr']
        print(f"Learning rates reset to config values: G={g_opt_cfg['lr']}, D={d_opt_cfg['lr']}")
    
    # Training loop
    print("Starting training...")
    
    train_cfg = config['train']
    total_steps = train_cfg['total_steps']
    batch_size = train_cfg['batch_size']
    
    # Learning rate schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=total_steps)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=total_steps)
    
    # LR decay milestones
    lr_decay_milestones = train_cfg.get('lr_decay_milestones', [])
    lr_decay_factor = train_cfg.get('lr_decay_factor', 0.5)
    
    # Early stopping
    early_stop_metric = train_cfg.get('early_stop_metric', 'kid')
    early_stop_patience = train_cfg.get('early_stop_patience', 3)
    early_stop_min_delta = train_cfg.get('early_stop_min_delta', 0.001)
    best_metric = float('inf') if early_stop_metric == 'kid' else float('-inf')
    patience_counter = 0
    
    # EMA checkpoint saving
    ema_checkpoint_every = train_cfg.get('ema_checkpoint_every', 2000)
    save_best_ema = train_cfg.get('save_best_ema', True)
    ema_metrics = []  # list of (step, metric_value) tuples
    
    augment_enabled = config['augment'].get('enabled', True)
    augment_policy = ','.join([k for k, v in config['augment']['policy'].items() if v])
    
    use_r1 = config['model']['d'].get('r1_reg_gamma', 0) > 0
    r1_gamma = config['model']['d'].get('r1_reg_gamma', 10.0)
    
    # Instance noise parameters
    instance_noise_std = 0.1
    instance_noise_decay = 20000  # Decay over 20k steps
    
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step)
    
    while step < total_steps:
        # Initialize loss tracking variables for logging
        d_loss = torch.tensor(0.0, device=device)
        g_gan_loss = torch.tensor(0.0, device=device)
        mi_loss_g = torch.tensor(0.0, device=device)
        fm_loss = torch.tensor(0.0, device=device)
        nan_recovery_active = False  # Track if NaN recovery happened this step
        
        for real_images in train_loader:
            if step >= total_steps:
                break
            
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            feature_matching_loss = torch.tensor(0.0, device=device)  # Initialize
            
            # ==================== Discriminator updates (variable count) ====================
            for _ in range(d_updates_per_step):
                discriminator.zero_grad()
                q_head.zero_grad()
                
                # Sample latents
                z = torch.randn(batch_size_actual, config['model']['z_dim'], device=device) * 0.8  # Truncated normal
                c_cat = sample_categorical(batch_size_actual, config['model']['c_cat_dim'], device)
                c_cont = sample_continuous(batch_size_actual, config['model']['c_cont_dim'], device=device) * 0.5  # Regularized
                
                with autocast('cuda', enabled=(scaler is not None)):
                    # Generate fake images
                    fake_images = generator(z, c_cat, c_cont)
                    
                    # Check for NaN in generator outputs
                    if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                        print(f"Warning: NaN/Inf in generator outputs at step {step}")
                        # NaN Recovery: Fix model parameters and skip this batch
                        fix_nan_parameters(generator)
                        nan_recovery_active = True
                        discriminator.zero_grad()
                        q_head.zero_grad()
                        continue
                    
                    # Apply DiffAugment to real and fake
                    if augment_enabled:
                        real_aug = DiffAugment(real_images, policy=augment_policy)
                        fake_aug = DiffAugment(fake_images.detach(), policy=augment_policy)
                    else:
                        real_aug = real_images
                        fake_aug = fake_images.detach()
                    
                    # Add instance noise (decays over time)
                    current_noise_std = instance_noise_std * max(0, 1 - step / instance_noise_decay)
                    if current_noise_std > 0:
                        real_aug = real_aug + torch.randn_like(real_aug) * current_noise_std
                        fake_aug = fake_aug + torch.randn_like(fake_aug) * current_noise_std
                    
                    # Discriminator outputs
                    real_logits, real_features = discriminator(real_aug, return_features=True)
                    fake_logits, fake_features = discriminator(fake_aug, return_features=True)
                    
                    # Check for NaN in discriminator outputs
                    if torch.isnan(real_logits).any() or torch.isinf(real_logits).any() or \
                       torch.isnan(fake_logits).any() or torch.isinf(fake_logits).any():
                        print(f"Warning: NaN/Inf in discriminator outputs at step {step}")
                        # NaN Recovery: Fix model parameters and skip this batch
                        fix_nan_parameters(discriminator)
                        fix_nan_parameters(q_head)
                        nan_recovery_active = True
                        generator.zero_grad()
                        discriminator.zero_grad()
                        q_head.zero_grad()
                        continue
                    
                    # Detach features for feature matching (don't need gradients through D)
                    real_features = real_features.detach()
                    
                    # GAN loss
                    d_loss = gan_loss_fn.dis_loss(real_logits, fake_logits)
                    
                    # InfoGAN loss (Q-head predicts codes from fake)
                    q_outputs = q_head(fake_features)
                    mi_loss, mi_stats = infogan_loss_fn(q_outputs, c_cat, c_cont)
                    
                    # Total D loss
                    mi_weight = config['loss'].get('mi_weight', 1.0)
                    d_total_loss = d_loss + mi_weight * mi_loss
                    
                    # R1 regularization (optional, applied less frequently)
                    if use_r1 and step % 16 == 0:
                        r1_loss = r1_regularization(discriminator, real_images)
                        d_total_loss = d_total_loss + r1_gamma * r1_loss
                
                # Check for NaN losses before backward
                if torch.isnan(d_total_loss) or torch.isinf(d_total_loss):
                    print(f"Warning: NaN/Inf detected in D loss at step {step}, d_total={d_total_loss.item() if torch.isfinite(d_total_loss) else 'nan'}")
                    # NaN Recovery: Reset gradients and fix model parameters
                    discriminator.zero_grad()
                    q_head.zero_grad()
                    fix_nan_parameters(discriminator)
                    fix_nan_parameters(q_head)
                    continue
                
                # Backward
                if scaler:
                    scaler.scale(d_total_loss).backward()
                    # More aggressive clipping during transfer learning recovery
                    clip_norm_d = 5.0 if step < RECOVERY_END else 10.0
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_norm_d)
                    torch.nn.utils.clip_grad_norm_(q_head.parameters(), max_norm=clip_norm_d)
                    scaler.step(d_optimizer)
                    scaler.update()
                else:
                    d_total_loss.backward()
                    # More aggressive clipping during transfer learning recovery
                    clip_norm_d = 5.0 if step < RECOVERY_END else 10.0
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_norm_d)
                    torch.nn.utils.clip_grad_norm_(q_head.parameters(), max_norm=clip_norm_d)
                    d_optimizer.step()
                
                # Note: Scheduler stepping moved outside the loop
            
            # decide recovery vs normal behavior per-step
            use_lsgan = (step < RECOVERY_END)
            current_g_ups = g_updates_per_step if step < RECOVERY_END else 1
            # during recovery, keep MI weight zero to avoid extra pressure
            mi_weight_step = 0.0 if step < RECOVERY_END else config['loss'].get('mi_weight', 1.0)
            
            # ==================== Generator updates (variable count) ====================
            feature_matching_weight = config['loss'].get('feature_matching_weight', 0.0)
            for _ in range(current_g_ups):
                generator.zero_grad()
                z = torch.randn(batch_size_actual, config['model']['z_dim'], device=device) * 0.8
                c_cat = sample_categorical(batch_size_actual, config['model']['c_cat_dim'], device)
                c_cont = sample_continuous(batch_size_actual, config['model']['c_cont_dim'], device=device) * 0.5

                with autocast('cuda', enabled=(scaler is not None)):
                    fake_images = generator(z, c_cat, c_cont)
                    
                    # Check for NaN in generator outputs
                    if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                        print(f"Warning: NaN/Inf in generator outputs at step {step}")
                        # NaN Recovery: Fix model parameters and skip this batch
                        fix_nan_parameters(generator)
                        generator.zero_grad()
                        continue
                    
                    fake_aug = DiffAugment(fake_images, policy=augment_policy) if augment_enabled else fake_images
                    fake_logits, fake_features = discriminator(fake_aug, return_features=True)

                    # choose loss type per-step
                    if use_lsgan:
                        g_gan_loss = (fake_logits - 1).pow(2).mean()
                    else:
                        g_gan_loss = gan_loss_fn.gen_loss(fake_logits)

                    # InfoGAN MI only after recovery
                    mi_loss_g = torch.tensor(0.0, device=device)
                    if not use_lsgan:
                        q_outputs = q_head(fake_features)
                        mi_loss_g, _ = infogan_loss_fn(q_outputs, c_cat, c_cont)

                    # Feature matching (global pooled)
                    fake_pooled = fake_features.mean(dim=[0, 2, 3])
                    real_pooled = real_features.mean(dim=[0, 2, 3])
                    
                    # Check for NaN in pooled features
                    if torch.isnan(fake_pooled).any() or torch.isinf(fake_pooled).any() or \
                       torch.isnan(real_pooled).any() or torch.isinf(real_pooled).any():
                        print(f"Warning: NaN/Inf in pooled features at step {step}")
                        fm_loss = torch.tensor(0.0, device=device)
                    else:
                        fm_loss = nn.L1Loss()(fake_pooled, real_pooled)

                    g_total_loss = g_gan_loss + mi_weight_step * mi_loss_g + feature_matching_weight * fm_loss
                
                # Check for NaN losses before backward
                if torch.isnan(g_total_loss) or torch.isinf(g_total_loss):
                    print(f"Warning: NaN/Inf detected in G loss at step {step}, g_total={g_total_loss.item() if torch.isfinite(g_total_loss) else 'nan'}")
                    # NaN Recovery: Reset gradients and fix model parameters
                    generator.zero_grad()
                    fix_nan_parameters(generator)
                    continue

                # backward & step
                if scaler:
                    scaler.scale(g_total_loss).backward()
                    # More aggressive clipping during transfer learning recovery
                    clip_norm_g = 2.0 if step < RECOVERY_END else 5.0
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_norm_g)
                    scaler.step(g_optimizer)
                    scaler.update()
                else:
                    g_total_loss.backward()
                    # Less aggressive clipping for generator to allow learning
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=3.0)
                    g_optimizer.step()

                # Note: Scheduler stepping moved outside the loop
            
            # Step schedulers once per global training step
            d_scheduler.step()
            g_scheduler.step()
            
            # Update EMA
            if ema:
                ema.update()
            
            # ==================== Logging ====================
            step += 1
            pbar.update(1)
            
            # LR decay at milestones
            if step in lr_decay_milestones:
                for g in g_optimizer.param_groups:
                    g['lr'] *= lr_decay_factor
                for d in d_optimizer.param_groups:
                    d['lr'] *= lr_decay_factor
                print(f"LR decayed at step {step}: G={g_optimizer.param_groups[0]['lr']:.6f}, D={d_optimizer.param_groups[0]['lr']:.6f}")
            
            if step % train_cfg['log_every'] == 0:
                # Show losses if computed, otherwise show recovery status
                if not nan_recovery_active and torch.isfinite(d_loss) and torch.isfinite(g_gan_loss) and torch.isfinite(mi_loss_g) and torch.isfinite(fm_loss):
                    pbar.set_postfix({
                        'D_loss': f"{d_loss.item():.4f}",
                        'G_loss': f"{g_gan_loss.item():.4f}",
                        'MI': f"{mi_loss_g.item():.4f}",
                        'FM': f"{fm_loss.item():.4f}",
                    })
                else:
                    status_msg = 'NaN detected (recovering)' if nan_recovery_active else 'Computing losses...'
                    pbar.set_postfix({
                        'status': status_msg,
                    })
            
            # ==================== Save Samples ====================
            if step % train_cfg['sample_every'] == 0:
                samples = generate_samples(generator, ema, config, num_samples=16, device=device, truncation_psi=0.8)
                if samples is not None:
                    sample_path = Path(config['paths']['samples']) / f'step_{step:07d}.png'
                    save_image(samples, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
                else:
                    print(f"Skipping sample save at step {step} due to NaN generation")
            
            # ==================== EMA Checkpoint Saving ====================
            if ema and step % ema_checkpoint_every == 0:
                ema_path = Path(config['paths']['ckpts']) / f'ema_step_{step:07d}.pt'
                torch.save({
                    'step': step,
                    'generator': generator.state_dict(),
                    'ema': ema.state_dict(),
                    'config': config,
                }, ema_path)
                print(f"EMA checkpoint saved: {ema_path}")
            
            # ==================== Evaluate Metrics ====================
            if step % train_cfg.get('eval_every', 10000) == 0 and step > 0:
                # TODO: Implement FID/KID evaluation with early stopping
                # For now, skip to avoid slowing down training
                pass
            
            # ==================== Early Stopping Check ====================
            # TODO: Implement metric-based early stopping
            
            if step % train_cfg['checkpoint_every'] == 0:
                save_checkpoint(
                    step, generator, discriminator, q_head,
                    g_optimizer, d_optimizer, ema, config,
                    f'checkpoint_step_{step:07d}.pt'
                )
                
                # Save latest checkpoint
                save_checkpoint(
                    step, generator, discriminator, q_head,
                    g_optimizer, d_optimizer, ema, config,
                    'checkpoint_latest.pt'
                )
                
                # Save EMA generator separately for inference
                if ema:
                    ema_path = Path(config['paths']['ckpts']) / 'ema_latest.pt'
                    torch.save({
                        'step': step,
                        'generator': generator.state_dict(),
                        'ema': ema.state_dict(),
                        'config': config,
                    }, ema_path)
    
    pbar.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
