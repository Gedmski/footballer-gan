"""
Evaluate FID and KID metrics for Footballer FaceGAN.
"""
import argparse
from pathlib import Path
import torch
from torch_fidelity import calculate_metrics
from tqdm import tqdm
from torchvision.utils import save_image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.generator import DCGANGenerator
from losses.infogan import sample_categorical, sample_continuous


def generate_fake_images(checkpoint_path, num_images, output_dir, device='cuda'):
    """Generate fake images from checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        print("Loading EMA weights...")
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
    
    # Generate images
    print(f"Generating {num_images} images...")
    batch_size = 50
    num_batches = (num_images + batch_size - 1) // batch_size
    
    img_idx = 0
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, num_images - img_idx)
            
            z = torch.randn(current_batch_size, model_cfg['z_dim'], device=device)
            c_cat = sample_categorical(current_batch_size, model_cfg['c_cat_dim'], device)
            c_cont = sample_continuous(current_batch_size, model_cfg['c_cont_dim'], device=device)
            
            fake_images = generator(z, c_cat, c_cont)
            
            # Save individual images
            for i in range(current_batch_size):
                img = fake_images[i]
                img_path = output_dir / f'{img_idx:05d}.png'
                save_image(img, img_path, normalize=True, value_range=(-1, 1))
                img_idx += 1
    
    print(f"Generated images saved to {output_dir}")


def compute_metrics(real_dir, fake_dir, metrics=['fid', 'kid']):
    """Compute FID and KID using torch-fidelity."""
    print(f"\nComputing metrics...")
    print(f"  Real images: {real_dir}")
    print(f"  Fake images: {fake_dir}")
    
    metrics_dict = calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=torch.cuda.is_available(),
        isc=False,
        fid='fid' in metrics,
        kid='kid' in metrics,
        prc=False,
        verbose=True,
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")
    print("="*50)
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate FID/KID for FaceGAN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/ema_latest.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--real',
        type=str,
        default='data/processed',
        help='Path to real images directory'
    )
    parser.add_argument(
        '--fake',
        type=str,
        default='outputs/samples_eval',
        help='Path to save/load fake images'
    )
    parser.add_argument(
        '--num_gen',
        type=int,
        default=5000,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--skip_generation',
        action='store_true',
        help='Skip generation and use existing fake images'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Generate fake images if needed
    if not args.skip_generation:
        generate_fake_images(
            args.checkpoint,
            args.num_gen,
            args.fake,
            args.device
        )
    else:
        print("Skipping image generation, using existing images...")
    
    # Compute metrics
    metrics = compute_metrics(args.real, args.fake, metrics=['fid', 'kid'])
    
    # Save results
    results_file = Path(args.fake).parent / 'metrics_results.txt'
    with open(results_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
