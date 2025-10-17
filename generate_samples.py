"""
Simple script to generate a few sample images to verify the model works.
Useful for quick testing after training completes.
"""
import sys
from pathlib import Path
import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.generator import DCGANGenerator
from losses.infogan import sample_categorical, sample_continuous


def generate_samples(checkpoint_path, num_samples=16, output_path='sample_grid.png'):
    """Generate a grid of sample faces."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    
    # Build generator
    model_cfg = config['model']
    generator = DCGANGenerator(
        z_dim=model_cfg['z_dim'],
        c_cat_dim=model_cfg['c_cat_dim'],
        c_cont_dim=model_cfg['c_cont_dim'],
        img_size=model_cfg['image_size'],
        base_channels=model_cfg['g']['base_channels'],
        out_channels=model_cfg['g']['out_channels'],
    ).to(device)
    
    # Load weights
    if 'ema' in ckpt and ckpt['ema'] is not None:
        print("Using EMA weights...")
        generator_state = {}
        for name, param in generator.named_parameters():
            if name in ckpt['ema']:
                generator_state[name] = ckpt['ema'][name]
            else:
                generator_state[name] = param.data
        generator.load_state_dict(generator_state, strict=False)
    else:
        generator.load_state_dict(ckpt['generator'])
    
    generator.eval()
    
    # Generate
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        z = torch.randn(num_samples, model_cfg['z_dim'], device=device)
        c_cat = sample_categorical(num_samples, model_cfg['c_cat_dim'], device)
        c_cont = sample_continuous(num_samples, model_cfg['c_cont_dim'], device=device)
        
        fake_images = generator(z, c_cat, c_cont)
    
    # Save
    save_image(fake_images, output_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved sample grid to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample images')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/ema_latest.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sample_grid.png',
        help='Output path for grid image'
    )
    args = parser.parse_args()
    
    generate_samples(args.checkpoint, args.num_samples, args.output)
