"""
Gradio interface for Footballer FaceGAN.
Loads EMA checkpoint and exposes latent sliders for c_cont, PCA, truncation, etc.
"""
import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import gradio as gr
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator import DCGANGenerator
from losses.infogan import sample_categorical, sample_continuous


class FaceGANApp:
    """Wrapper for FaceGAN Gradio interface."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.config = ckpt['config']
        
        # Build generator
        model_cfg = self.config['model']
        self.generator = DCGANGenerator(
            z_dim=model_cfg['z_dim'],
            c_cat_dim=model_cfg['c_cat_dim'],
            c_cont_dim=model_cfg['c_cont_dim'],
            img_size=model_cfg['image_size'],
            base_channels=model_cfg['g']['base_channels'],
            out_channels=model_cfg['g']['out_channels'],
        ).to(device)
        
        # Load weights (use EMA if available)
        if 'ema' in ckpt and ckpt['ema'] is not None:
            print("Loading EMA weights...")
            # EMA stores shadow weights dict
            generator_state = {}
            for name, param in self.generator.named_parameters():
                if name in ckpt['ema']:
                    generator_state[name] = ckpt['ema'][name]
                else:
                    generator_state[name] = param.data
            self.generator.load_state_dict(generator_state, strict=False)
        else:
            self.generator.load_state_dict(ckpt['generator'])
        
        self.generator.eval()
        
        self.z_dim = model_cfg['z_dim']
        self.c_cat_dim = model_cfg['c_cat_dim']
        self.c_cont_dim = model_cfg['c_cont_dim']
        
        print("Model loaded successfully!")
    
    def generate(self, seed, c_cat_idx, c_cont_0, c_cont_1, c_cont_2, truncation_psi=1.0):
        """Generate image from latent parameters."""
        torch.manual_seed(seed)
        
        with torch.no_grad():
            # Sample latent
            z = torch.randn(1, self.z_dim, device=self.device) * truncation_psi
            
            # Categorical code (one-hot)
            c_cat = torch.zeros(1, self.c_cat_dim, device=self.device)
            c_cat[0, c_cat_idx] = 1.0
            
            # Continuous codes
            c_cont = torch.tensor([[c_cont_0, c_cont_1, c_cont_2]], device=self.device)
            
            # Generate
            fake_img = self.generator(z, c_cat, c_cont)
            
            # Convert to PIL image
            fake_img = fake_img.squeeze(0).cpu()
            fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
            fake_img = (fake_img * 255).byte().permute(1, 2, 0).numpy()
            
            return Image.fromarray(fake_img)
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="⚽ Footballer FaceGAN") as demo:
            gr.Markdown("# ⚽ Footballer FaceGAN Explorer")
            gr.Markdown(
                "Interactive latent space exploration for footballer face generation. "
                "Adjust sliders to control facial attributes."
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎲 Generation Controls")
                    
                    seed_slider = gr.Slider(
                        minimum=0, maximum=999999, value=42, step=1,
                        label="Random Seed"
                    )
                    
                    truncation_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Truncation ψ (1.0 = full diversity)"
                    )
                    
                    gr.Markdown("### 🏷️ Categorical Code")
                    c_cat_dropdown = gr.Dropdown(
                        choices=list(range(self.c_cat_dim)),
                        value=0,
                        label=f"Category (0-{self.c_cat_dim - 1})"
                    )
                    
                    gr.Markdown("### 🎨 Continuous Codes")
                    c_cont_0 = gr.Slider(
                        minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                        label="c_cont[0] (e.g., lighting)"
                    )
                    c_cont_1 = gr.Slider(
                        minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                        label="c_cont[1] (e.g., complexion)"
                    )
                    c_cont_2 = gr.Slider(
                        minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                        label="c_cont[2] (e.g., face shape)"
                    )
                    
                    generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Face", type="pil")
            
            # Set up event handler
            generate_btn.click(
                fn=self.generate,
                inputs=[
                    seed_slider, c_cat_dropdown,
                    c_cont_0, c_cont_1, c_cont_2, truncation_slider
                ],
                outputs=output_image
            )
            
            # Examples
            gr.Markdown("### 📚 Example Configurations")
            gr.Examples(
                examples=[
                    [42, 0, 0.0, 0.0, 0.0, 1.0],
                    [123, 3, 1.0, -0.5, 0.5, 0.8],
                    [456, 7, -1.0, 1.0, 0.0, 1.2],
                    [789, 2, 0.5, 0.5, -1.0, 0.7],
                ],
                inputs=[
                    seed_slider, c_cat_dropdown,
                    c_cont_0, c_cont_1, c_cont_2, truncation_slider
                ],
                outputs=output_image,
                fn=self.generate,
                cache_examples=False,
            )
        
        return demo


def main():
    parser = argparse.ArgumentParser(description='Footballer FaceGAN Gradio App')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/ema_latest.pt',
        help='Path to checkpoint file'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--share', action='store_true', help='Create public share link')
    parser.add_argument('--port', type=int, default=7860, help='Port number')
    args = parser.parse_args()
    
    # Create app
    app = FaceGANApp(args.checkpoint, device=args.device)
    demo = app.create_interface()
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
