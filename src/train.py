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
                    d_optimizer, ema=None, device='cuda'):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
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


def generate_samples(generator, config, num_samples=16, device='cuda'):
    """Generate sample images."""
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, config['model']['z_dim'], device=device)
        c_cat = sample_categorical(num_samples, config['model']['c_cat_dim'], device)
        c_cont = sample_continuous(num_samples, config['model']['c_cont_dim'], device=device)
        
        fake_images = generator(z, c_cat, c_cont)
    
    generator.train()
    return fake_images


def train():
    """Main training loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Footballer FaceGAN')
    parser.add_argument('--config', type=str, default='configs/dcgan_infogan_128.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
    
    # AMP scaler
    scaler = GradScaler('cuda') if config.get('amp', True) and device == 'cuda' else None
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, generator, discriminator, q_head,
            g_optimizer, d_optimizer, ema, device
        )
    
    # Training loop
    print("Starting training...")
    
    train_cfg = config['train']
    total_steps = train_cfg['total_steps']
    batch_size = train_cfg['batch_size']
    
    augment_enabled = config['augment'].get('enabled', True)
    augment_policy = ','.join([k for k, v in config['augment']['policy'].items() if v])
    
    use_r1 = config['model']['d'].get('r1_reg_gamma', 0) > 0
    r1_gamma = config['model']['d'].get('r1_reg_gamma', 10.0)
    
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step)
    
    while step < total_steps:
        for real_images in train_loader:
            if step >= total_steps:
                break
            
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # ==================== Train Discriminator ====================
            discriminator.zero_grad()
            q_head.zero_grad()
            
            # Sample latents
            z = torch.randn(batch_size_actual, config['model']['z_dim'], device=device)
            c_cat = sample_categorical(batch_size_actual, config['model']['c_cat_dim'], device)
            c_cont = sample_continuous(batch_size_actual, config['model']['c_cont_dim'], device=device)
            
            with autocast('cuda', enabled=(scaler is not None)):
                # Generate fake images
                fake_images = generator(z, c_cat, c_cont)
                
                # Apply DiffAugment to real and fake
                if augment_enabled:
                    real_aug = DiffAugment(real_images, policy=augment_policy)
                    fake_aug = DiffAugment(fake_images.detach(), policy=augment_policy)
                else:
                    real_aug = real_images
                    fake_aug = fake_images.detach()
                
                # Discriminator outputs
                real_logits = discriminator(real_aug)
                fake_logits, fake_features = discriminator(fake_aug, return_features=True)
                
                # GAN loss
                d_loss = gan_loss_fn.dis_loss(real_logits, fake_logits)
                
                # InfoGAN loss (Q-head predicts codes from fake)
                q_outputs = q_head(fake_features)
                mi_loss, mi_stats = infogan_loss_fn(q_outputs, c_cat, c_cont)
                
                # Total D loss
                d_total_loss = d_loss + mi_loss
                
                # R1 regularization (optional, applied less frequently)
                if use_r1 and step % 16 == 0:
                    r1_loss = r1_regularization(discriminator, real_images)
                    d_total_loss = d_total_loss + r1_gamma * r1_loss
            
            # Backward
            if scaler:
                scaler.scale(d_total_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                d_total_loss.backward()
                d_optimizer.step()
            
            # ==================== Train Generator ====================
            generator.zero_grad()
            
            # Sample new latents
            z = torch.randn(batch_size_actual, config['model']['z_dim'], device=device)
            c_cat = sample_categorical(batch_size_actual, config['model']['c_cat_dim'], device)
            c_cont = sample_continuous(batch_size_actual, config['model']['c_cont_dim'], device=device)
            
            with autocast('cuda', enabled=(scaler is not None)):
                # Generate fake images
                fake_images = generator(z, c_cat, c_cont)
                
                # Apply DiffAugment
                if augment_enabled:
                    fake_aug = DiffAugment(fake_images, policy=augment_policy)
                else:
                    fake_aug = fake_images
                
                # Discriminator outputs
                fake_logits, fake_features = discriminator(fake_aug, return_features=True)
                
                # GAN loss
                g_loss = gan_loss_fn.gen_loss(fake_logits)
                
                # InfoGAN loss
                q_outputs = q_head(fake_features)
                mi_loss_g, mi_stats_g = infogan_loss_fn(q_outputs, c_cat, c_cont)
                
                # Total G loss
                g_total_loss = g_loss + mi_loss_g
            
            # Backward
            if scaler:
                scaler.scale(g_total_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                g_total_loss.backward()
                g_optimizer.step()
            
            # Update EMA
            if ema:
                ema.update()
            
            # ==================== Logging ====================
            step += 1
            pbar.update(1)
            
            if step % train_cfg['log_every'] == 0:
                pbar.set_postfix({
                    'D_loss': f"{d_loss.item():.4f}",
                    'G_loss': f"{g_loss.item():.4f}",
                    'MI': f"{mi_loss.item():.4f}",
                })
            
            # ==================== Save Samples ====================
            if step % train_cfg['sample_every'] == 0:
                samples = generate_samples(generator, config, num_samples=16, device=device)
                sample_path = Path(config['paths']['samples']) / f'step_{step:07d}.png'
                save_image(samples, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
            
            # ==================== Save Checkpoint ====================
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
