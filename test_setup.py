"""
Test script to verify the installation and setup.
Run this before training to ensure everything works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("FOOTBALLER FACEGAN - INSTALLATION TEST")
print("=" * 60)

# Test 1: Import core libraries
print("\n[1/8] Testing core library imports...")
try:
    import torch
    import torchvision
    import numpy as np
    import yaml
    import gradio as gr
    from tqdm import tqdm
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ Torchvision {torchvision.__version__}")
    print(f"✓ NumPy {np.__version__}")
    print(f"✓ Gradio {gr.__version__}")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: CUDA availability
print("\n[2/8] Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ CUDA not available (will use CPU - very slow)")

# Test 3: Project structure
print("\n[3/8] Verifying project structure...")
required_dirs = [
    'src/models',
    'src/datasets',
    'src/losses',
    'src/augment',
    'src/viz',
    'src/app',
    'configs',
    'data/raw',
    'data/processed',
    'outputs/checkpoints',
    'outputs/samples',
    'outputs/logs',
    'reports',
]

for dir_path in required_dirs:
    if not Path(dir_path).exists():
        print(f"✗ Missing directory: {dir_path}")
        sys.exit(1)

print("✓ All required directories exist")

# Test 4: Import project modules
print("\n[4/8] Testing project module imports...")
try:
    from models.generator import build_generator, DCGANGenerator, EMAWrapper
    from models.discriminator import build_discriminator, DCGANDiscriminator
    from models.q_head import build_q_head, QHead
    from losses.gan_losses import build_gan_loss, GANLoss
    from losses.infogan import build_infogan_loss, InfoGANLoss
    from losses.infogan import sample_categorical, sample_continuous
    from augment.diffaugment import DiffAugment
    from datasets.fm_cutout import FM23CutoutDataset
    print("✓ All project modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing project modules: {e}")
    sys.exit(1)

# Test 5: Load config
print("\n[5/8] Loading configuration...")
try:
    config_path = Path('configs/dcgan_infogan_128.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    print(f"  Project: {config['project_name']}")
    print(f"  Image size: {config['model']['image_size']}")
    print(f"  Batch size: {config['train']['batch_size']}")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)

# Test 6: Build models
print("\n[6/8] Building models (dry run)...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    q_head = build_q_head(config, discriminator.feature_dim)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    q_params = sum(p.numel() for p in q_head.parameters())
    
    print(f"✓ Generator: {g_params:,} parameters")
    print(f"✓ Discriminator: {d_params:,} parameters")
    print(f"✓ Q-Head: {q_params:,} parameters")
    print(f"  Total: {g_params + d_params + q_params:,} parameters")
except Exception as e:
    print(f"✗ Error building models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Forward pass
print("\n[7/8] Testing forward pass...")
try:
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    q_head = q_head.to(device)
    
    batch_size = 4
    z = torch.randn(batch_size, config['model']['z_dim'], device=device)
    c_cat = sample_categorical(batch_size, config['model']['c_cat_dim'], device)
    c_cont = sample_continuous(batch_size, config['model']['c_cont_dim'], device=device)
    
    # Generator forward
    fake_images = generator(z, c_cat, c_cont)
    assert fake_images.shape == (batch_size, 3, 128, 128), "Generator output shape mismatch"
    
    # Discriminator forward
    logits, features = discriminator(fake_images, return_features=True)
    assert logits.shape == (batch_size,), "Discriminator output shape mismatch"
    
    # Q-head forward
    q_outputs = q_head(features)
    assert 'cat_logits' in q_outputs, "Q-head missing categorical output"
    assert 'cont_mean' in q_outputs, "Q-head missing continuous output"
    
    print(f"✓ Forward pass successful")
    print(f"  Generated image shape: {fake_images.shape}")
    print(f"  Value range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Loss computation
print("\n[8/8] Testing loss computation...")
try:
    gan_loss = build_gan_loss(config)
    infogan_loss = build_infogan_loss(config)
    
    # Dummy discriminator outputs
    real_logits = torch.randn(batch_size, device=device)
    fake_logits = torch.randn(batch_size, device=device)
    
    # Compute GAN losses
    d_loss = gan_loss.dis_loss(real_logits, fake_logits)
    g_loss = gan_loss.gen_loss(fake_logits)
    
    # Compute InfoGAN loss
    mi_loss, mi_stats = infogan_loss(q_outputs, c_cat, c_cont)
    
    print(f"✓ Loss computation successful")
    print(f"  D_loss: {d_loss.item():.4f}")
    print(f"  G_loss: {g_loss.item():.4f}")
    print(f"  MI_loss: {mi_loss.item():.4f}")
except Exception as e:
    print(f"✗ Error in loss computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYou're ready to train!")
print("\nNext steps:")
print("  1. Place images in data/raw/")
print("  2. Run: python src/preprocess_data.py")
print("  3. Run: python src/train.py")
print("\nFor more details, see QUICKSTART.md")
print("=" * 60)
