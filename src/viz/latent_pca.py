"""
Latent space PCA analysis for Footballer FaceGAN.
Visualize principal components of the latent space.
"""
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator import DCGANGenerator
from losses.infogan import sample_categorical, sample_continuous


def collect_latents(generator, config, num_samples=10000, device='cuda'):
    """
    Collect random latent vectors and their generated images.
    
    Returns:
        latents: (num_samples, latent_dim) numpy array
    """
    generator.eval()
    
    z_dim = config['model']['z_dim']
    c_cat_dim = config['model']['c_cat_dim']
    c_cont_dim = config['model']['c_cont_dim']
    
    latents_list = []
    
    print(f"Collecting {num_samples} latent samples...")
    with torch.no_grad():
        for _ in tqdm(range(num_samples // 100)):
            z = torch.randn(100, z_dim, device=device)
            c_cat = sample_categorical(100, c_cat_dim, device)
            c_cont = sample_continuous(100, c_cont_dim, device=device)
            
            # Concatenate all latent components
            latent = torch.cat([z, c_cat, c_cont], dim=1)  # (100, z_dim + c_cat_dim + c_cont_dim)
            
            latents_list.append(latent.cpu().numpy())
    
    latents = np.concatenate(latents_list, axis=0)
    print(f"Collected latents shape: {latents.shape}")
    
    return latents


def perform_pca(latents, n_components=10):
    """
    Perform PCA on latent vectors.
    
    Returns:
        pca: Fitted PCA object
        transformed: PCA-transformed latents
    """
    print(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(latents)
    
    print("PCA explained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    return pca, transformed


def visualize_pca(pca, transformed, output_dir):
    """Visualize PCA results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
           pca.explained_variance_ratio_ * 100)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_explained_variance.png', dpi=150)
    plt.close()
    
    # Plot cumulative explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('Cumulative Explained Variance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_cumulative_variance.png', dpi=150)
    plt.close()
    
    # Plot 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5, s=1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent Space Projection (PC1 vs PC2)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_2d_projection.png', dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def save_pca_model(pca, config, output_path):
    """Save PCA model for use in Gradio app."""
    output_path = Path(output_path)
    
    torch.save({
        'pca_components': pca.components_,
        'pca_mean': pca.mean_,
        'pca_explained_variance': pca.explained_variance_,
        'pca_explained_variance_ratio': pca.explained_variance_ratio_,
        'config': config,
    }, output_path)
    
    print(f"PCA model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Latent Space PCA Analysis')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/ema_latest.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of latent samples'
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=10,
        help='Number of PCA components'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/pca_analysis',
        help='Output directory'
    )
    args = parser.parse_args()
    
    device = args.device
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        generator_state = {}
        for name, param in generator.named_parameters():
            if name in ckpt['ema']:
                generator_state[name] = ckpt['ema'][name]
            else:
                generator_state[name] = param.data
        generator.load_state_dict(generator_state, strict=False)
    else:
        generator.load_state_dict(ckpt['generator'])
    
    # Collect latents
    latents = collect_latents(generator, config, args.num_samples, device)
    
    # Perform PCA
    pca, transformed = perform_pca(latents, args.n_components)
    
    # Visualize
    visualize_pca(pca, transformed, args.output_dir)
    
    # Save PCA model
    save_pca_model(pca, config, Path(args.output_dir) / 'pca_model.pt')
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
