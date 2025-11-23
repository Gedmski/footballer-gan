"""
Train a DCGAN + InfoGAN on footballer face dataset.
Simplified version with stabilizers for GAN training imbalance.
"""
import os
import argparse
from pathlib import Path
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def debug_latent_codes(config, device='cuda'):
    """Debug InfoGAN latent code generation."""
    print("\n=== Latent Codes Debug ===")
    
    z = torch.randn(4, config['model']['z_dim'], device=device)
    c_cat = sample_categorical(4, config['model']['c_cat_dim'], device)
    c_cont = sample_continuous(4, config['model']['c_cont_dim'], device=device)
    
    print(f"z shape: {z.shape}, mean: {z.mean().item():.4f}, std: {z.std().item():.4f}")
    print(f"c_cat shape: {c_cat.shape}, unique values: {c_cat.unique().tolist()}")
    print(f"c_cont shape: {c_cont.shape}, range: [{c_cont.min().item():.4f}, {c_cont.max().item():.4f}]")
    
    # Concatenated latent
    latent = torch.cat([z, c_cat, c_cont], dim=1)
    print(f"latent shape: {latent.shape}, std per dim: {latent.std(dim=0).mean().item():.4f}")
    
    print("=== End Latent Codes Debug ===\n")


def debug_generator_forward(generator, config, device='cuda'):
    """Debug generator forward pass to find where gradients vanish."""
    generator.train()  # Make sure it's in train mode
    z_dim = config['model']['z_dim']
    
    print("\n=== Generator Debug ===")
    
    # Test with different z values
    z1 = torch.randn(1, z_dim, device=device).requires_grad_(True)
    z2 = torch.randn(1, z_dim, device=device).requires_grad_(True)
    
    c_cat = sample_categorical(1, config['model']['c_cat_dim'], device)
    c_cont = sample_continuous(1, config['model']['c_cont_dim'], device=device)
    
    # Forward pass 1
    x1 = generator(z1, c_cat, c_cont)
    loss1 = x1.mean()
    loss1.backward(retain_graph=True)
    grad1 = z1.grad.norm().item()
    z1.grad.zero_()
    
    # Forward pass 2  
    x2 = generator(z2, c_cat, c_cont)
    loss2 = x2.mean()
    loss2.backward(retain_graph=True)
    grad2 = z2.grad.norm().item()
    z2.grad.zero_()
    
    print(f"z1 output mean/std: {x1.mean().item():.6f}, {x1.std().item():.6f}, grad_norm: {grad1:.6f}")
    print(f"z2 output mean/std: {x2.mean().item():.6f}, {x2.std().item():.6f}, grad_norm: {grad2:.6f}")
    
    # Difference
    diff = (x1 - x2).abs().mean().item()
    print(f"Mean difference between z1/z2 outputs: {diff:.6f}")
    
    # Check BatchNorm running stats
    for name, module in generator.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"{name}: running_mean={module.running_mean.mean().item():.6f}, running_var={module.running_var.mean().item():.6f}")
    
    print("=== End Generator Debug ===\n")


def check_generator_collapse(generator, config, device='cuda'):
    """Quick diagnostics to detect gray collapse."""
    generator.eval()
    z_dim = config['model']['z_dim']
    
    print("\n=== Collapse Diagnostics ===")
    
    # Check if G uses z (Jacobian norm)
    z = torch.randn(8, z_dim, device=device).requires_grad_(True)
    c_cat = sample_categorical(8, config['model']['c_cat_dim'], device)
    c_cont = sample_continuous(8, config['model']['c_cont_dim'], device=device)
    
    x = generator(z, c_cat, c_cont)
    s = x.mean()
    jac_norm = torch.autograd.grad(s, z, retain_graph=False, create_graph=False)[0].norm().item()
    print(f"Jacobian norm wrt z: {jac_norm:.6f} (should be > 0.0; ~1-10 typical)")
    
    # Check batch diversity
    with torch.no_grad():
        z_batch = torch.randn(32, z_dim, device=device)
        c_cat_batch = sample_categorical(32, config['model']['c_cat_dim'], device)
        c_cont_batch = sample_continuous(32, config['model']['c_cont_dim'], device=device)
        imgs = generator(z_batch, c_cat_batch, c_cont_batch)
        
        print(f"Per-tensor mean/std: {imgs.mean().item():.6f}, {imgs.std().item():.6f}")
        inter_sample_std = imgs.view(32, -1).std(dim=1).mean().item()
        print(f"Inter-sample std: {inter_sample_std:.6f} (should be > 0.05 for diversity)")
    
    generator.train()
    print("=== End Diagnostics ===\n")


def check_ema_sanity(generator, ema, config, device='cuda'):
    """Check if EMA is working correctly (not causing gray outputs)."""
    if ema is None:
        print("No EMA to check")
        return
        
    generator.eval()
    z_dim = config['model']['z_dim']
    
    print("\n=== EMA Sanity Check ===")
    
    with torch.no_grad():
        z = torch.randn(16, z_dim, device=device)
        c_cat = sample_categorical(16, config['model']['c_cat_dim'], device)
        c_cont = sample_continuous(16, config['model']['c_cont_dim'], device=device)
        
        # Raw generator output
        x_raw = generator(z, c_cat, c_cont)
        
        # EMA output (apply shadow temporarily)
        ema.apply_shadow()
        x_ema = generator(z, c_cat, c_cont)
        ema.restore()
        
        print(f"Raw G std: {x_raw.std().item():.6f}")
        print(f"EMA G std: {x_ema.std().item():.6f}")
        
        if x_raw.std().item() > 0.05 and x_ema.std().item() < 0.01:
            print("WARNING: EMA may not be copying BatchNorm buffers - sampling from EMA gives gray outputs!")
        elif x_ema.std().item() > 0.05:
            print("EMA appears to be working correctly")
        else:
            print("WARNING: Both raw and EMA outputs have low diversity - deeper issue")
    
    generator.train()
    print("=== End EMA Check ===\n")


def create_directories(config):
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


def generate_samples(generator, ema, config, num_samples=16, device='cuda'):
    """Generate sample images using EMA with truncation."""
    # If EMA wrapper is provided, temporarily apply its shadow weights for sampling
    use_ema = ema is not None
    if use_ema:
        ema.apply_shadow()

    # Use the raw generator (EMA shadow has been applied onto its params if requested)
    generator.eval()
    with torch.no_grad():
        # Truncated normal sampling (ψ = 0.8)
        z = torch.randn(num_samples, config['model']['z_dim'], device=device) * 0.8
        c_cat = sample_categorical(num_samples, config['model']['c_cat_dim'], device)
        c_cont = sample_continuous(num_samples, config['model']['c_cont_dim'], device=device)

        fake_images = generator(z, c_cat, c_cont)

    generator.train()

    if use_ema:
        ema.restore()

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

    # Run collapse diagnostics
    debug_latent_codes(config, device)
    check_generator_collapse(generator, config, device)
    debug_generator_forward(generator, config, device)
    check_ema_sanity(generator, ema, config, device)

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

    # --- Stabilizers ---
    for g in g_optimizer.param_groups: g["lr"] = 2e-4  # Higher G LR for recovery
    for d in d_optimizer.param_groups: d["lr"] = 3e-5   # Lower D LR to let G catch up
    
    EMA_DECAY = 0.9995
    R1_GAMMA = 10.0
    G_UPS = 2        # 2 G steps per 1 D step for recovery
    REAL_LABEL = 1.0 # Turn OFF label smoothing
    CLIP_NORM = 5.0
    USE_LSGAN_RECOVERY = True  # Use LSGAN for recovery
    
    def instance_noise_sigma(step, warmdown=8000, start=0.02):
        # DISABLED during recovery - return 0
        return 0.0  # Turn off instance noise for recovery    # Resume from checkpoint if specified
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
    # gradient accumulation support (emulate larger batch sizes)
    grad_accum = int(train_cfg.get('grad_accum', 1))
    g_accum_count = 0

    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step)

    while step < total_steps:
        for real_images in train_loader:
            if step >= total_steps:
                break

            real_images = real_images.to(device)
            bsz = real_images.size(0)
            z_dim = config['model']['z_dim']

            # Ensure generator is in training mode
            generator.train()
            for _ in range(G_UPS):
                z = torch.randn(bsz, z_dim, device=device)
                fake = generator(z, 
                               sample_categorical(bsz, config['model']['c_cat_dim'], device),
                               sample_continuous(bsz, config['model']['c_cont_dim'], device=device))

                # G loss with LSGAN recovery + feature matching
                if USE_LSGAN_RECOVERY:
                    g_gan_loss = (discriminator(fake) - 1).pow(2).mean()
                else:
                    g_gan_loss = F.binary_cross_entropy_with_logits(discriminator(fake), torch.ones_like(discriminator(fake)))

                # Feature matching loss (using global average pooled features)
                with torch.no_grad():
                    f_real = discriminator.features(real_images).mean(dim=[0, 2, 3])  # (C,)
                f_fake = discriminator.features(fake).mean(dim=[0, 2, 3])  # (C,)
                fm_loss = (f_fake - f_real).abs().mean()

                g_loss = g_gan_loss + 10.0 * fm_loss

                # Gradient accumulation: scale loss and accumulate gradients
                if g_accum_count == 0:
                    g_optimizer.zero_grad(set_to_none=True)
                (g_loss / float(grad_accum)).backward()
                g_accum_count += 1
                if g_accum_count >= grad_accum:
                    g_optimizer.step()
                    # Update EMA after optimizer step
                    if ema:
                        ema.update()
                    g_accum_count = 0
            
            # ---- Discriminator update ----
            with torch.no_grad():
                fake_detached = generator(z,
                                        sample_categorical(bsz, config['model']['c_cat_dim'], device),
                                        sample_continuous(bsz, config['model']['c_cont_dim'], device=device))

            # instance noise (to both real and fake)
            sigma = instance_noise_sigma(step)
            if sigma > 0:
                real_noisy = real_images + torch.randn_like(real_images) * sigma
                fake_noisy = fake_detached + torch.randn_like(fake_detached) * sigma
            else:
                real_noisy, fake_noisy = real_images, fake_detached

            d_real = discriminator(real_noisy)
            d_fake = discriminator(fake_noisy)
            
            # D loss with LSGAN recovery (no label smoothing)
            if USE_LSGAN_RECOVERY:
                d_loss_real = (d_real - 1).pow(2).mean()
                d_loss_fake = (d_fake).pow(2).mean()
            else:
                d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            
            d_loss = d_loss_real + d_loss_fake            # optional R1 every 16 steps
            if step % 16 == 0:
                real_noisy.requires_grad_(True)
                d_real_r1 = discriminator(real_noisy)
                grad = torch.autograd.grad(d_real_r1.sum(), real_noisy, create_graph=True)[0]
                r1_penalty = (grad.flatten(1).pow(2).sum(1)).mean()
                d_loss = d_loss + R1_GAMMA * 0.5 * r1_penalty

            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), CLIP_NORM)
            d_optimizer.step()

            # ==================== Logging ====================
            step += 1
            pbar.update(1)

            if step % train_cfg['log_every'] == 0:
                pbar.set_postfix({
                    'D_loss': f"{d_loss.item():.4f}",
                    'G_loss': f"{g_loss.item():.4f}",
                    'MI': f"0.0000",  # Disabled
                })

            # ==================== Save Samples ====================
            if step % train_cfg['sample_every'] == 0:
                samples = generate_samples(generator, ema, config, num_samples=16, device=device)
                
                # Quick diversity check before saving
                sample_std = samples.std().item()
                sample_mean = samples.mean().item()
                print(f"Sample diversity check - mean: {sample_mean:.4f}, std: {sample_std:.4f}")
                
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


def debug_only():
    """Run diagnostics only, without training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Debug Footballer FaceGAN')
    parser.add_argument('--config', type=str, default='configs/dcgan_infogan_128.yaml',
                        help='Path to config file')
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

    # Build models
    print("Building models...")
    generator = build_generator(config).to(device)

    # Build EMA wrapper for generator
    ema = None
    if config['model']['g'].get('use_ema', True):
        ema = EMAWrapper(generator, decay=config['model']['g'].get('ema_decay', 0.999))

    # Run diagnostics
    debug_latent_codes(config, device)
    check_generator_collapse(generator, config, device)
    debug_generator_forward(generator, config, device)
    check_ema_sanity(generator, ema, config, device)


def debug_only():
    """Run diagnostics only, without training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Debug Footballer FaceGAN')
    parser.add_argument('--config', type=str, default='configs/dcgan_infogan_128.yaml',
                        help='Path to config file')
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

    # Build models
    print("Building models...")
    generator = build_generator(config).to(device)

    # Build EMA wrapper for generator
    ema = None
    if config['model']['g'].get('use_ema', True):
        ema = EMAWrapper(generator, decay=config['model']['g'].get('ema_decay', 0.999))

    # Run diagnostics
    debug_latent_codes(config, device)
    check_generator_collapse(generator, config, device)
    debug_generator_forward(generator, config, device)
    check_ema_sanity(generator, ema, config, device)


if __name__ == "__main__":
    import sys
    if '--debug' in sys.argv:
        # Remove --debug from sys.argv so argparse doesn't complain
        sys.argv.remove('--debug')
        debug_only()
    else:
        train()