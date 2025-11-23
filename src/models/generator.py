"""
Generator network for Footballer FaceGAN.
Upsamples latent z + InfoGAN codes to 128x128 RGB image.
DCGAN-style architecture with ConvTranspose2d layers.
"""
import torch
import torch.nn as nn
import copy


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator with InfoGAN conditioning.
    
    Architecture:
        Input: [z (64), c_cat (8), c_cont (3)] -> concat to latent_dim
        Project to (ngf*16, 4, 4)
        ConvTranspose blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        Output: (3, 128, 128) RGB image in [-1, 1]
    
    Args:
        z_dim: Gaussian latent dimension
        c_cat_dim: Categorical code dimension
        c_cont_dim: Continuous code dimension
        img_size: Output image size (128)
        base_channels: Base channel multiplier (ngf)
        out_channels: Output channels (3 for RGB)
    """
    
    def __init__(
        self,
        z_dim=64,
        c_cat_dim=8,
        c_cont_dim=3,
        img_size=128,
        base_channels=64,
        out_channels=3,
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.c_cat_dim = c_cat_dim
        self.c_cont_dim = c_cont_dim
        self.latent_dim = z_dim + c_cat_dim + c_cont_dim
        
        ngf = base_channels
        
        # Calculate number of upsample blocks needed
        # 128 = 4 * 2^5, so we need 5 upsample blocks
        assert img_size in [64, 128, 192, 256], "Only 64, 128, 192, 256 supported"
        
        if img_size == 128:
            n_up = 5  # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        elif img_size == 64:
            n_up = 4  # 4 -> 8 -> 16 -> 32 -> 64
        elif img_size == 192:
            n_up = 5  # 4 -> 8 -> 16 -> 32 -> 64 -> 128 (same as 128, but final upsample goes to 192)
        else:  # 256
            n_up = 6
        
        # Initial projection: latent -> (ngf*16, 4, 4)
        self.init_size = 4
        self.project = nn.Sequential(
            nn.Linear(self.latent_dim, ngf * 16 * self.init_size * self.init_size),
            # nn.BatchNorm1d(ngf * 16 * self.init_size * self.init_size),  # Remove problematic BN1d
            nn.ReLU(True),
        )
        
        # Upsample blocks
        layers = []
        in_ch = ngf * 16
        
        for i in range(n_up):
            out_ch = in_ch // 2 if i < n_up - 1 else ngf
            # Use anti-aliasing upsampling for the final 128->256 block to avoid checkerboard artifacts
            # Also skip BatchNorm for the final block to prevent NaN issues - use InstanceNorm instead
            use_transpose = not (img_size == 256 and i == n_up - 1)  # Last block for 256x256 uses Upsample+Conv
            use_bn = not (img_size == 256 and i == n_up - 1)  # Skip BN for final 256 block
            layers.append(
                self._make_upsample_block(in_ch, out_ch, use_bn=use_bn, use_transpose=use_transpose, use_instance_norm=(img_size == 256 and i == n_up - 1))
            )
            in_ch = out_ch
        
        # Final conv to RGB (no upsampling, just channel change)
        self.to_rgb = nn.Conv2d(ngf, out_channels, 3, 1, 1, bias=False)
        layers.append(self.to_rgb)
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        
        # Small random init for to_rgb instead of zeros to avoid dead neurons
        if img_size == 256:
            nn.init.normal_(self.to_rgb.weight, 0.0, 0.01)  # Small random weights
            if self.to_rgb.bias is not None:
                nn.init.zeros_(self.to_rgb.bias)
    
    def _make_upsample_block(self, in_ch, out_ch, use_bn=True, use_transpose=True, use_instance_norm=False):
        """Create upsample block. Use Upsample+Conv for final 128->256 to avoid checkerboard artifacts."""
        if use_transpose:
            # Standard DCGAN upsampling with ConvTranspose2d
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))
        else:
            # Anti-aliasing upsampling: Upsample + Conv (for final 128->256)
            layers = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            ]
            if use_instance_norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            elif use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Use LeakyReLU like suggested
        
        return nn.Sequential(*layers)
    
    def forward(self, z, c_cat=None, c_cont=None):
        """
        Args:
            z: (B, z_dim) Gaussian noise
            c_cat: (B, c_cat_dim) Categorical codes (one-hot or soft)
            c_cont: (B, c_cont_dim) Continuous codes
        
        Returns:
            img: (B, 3, 128, 128) Generated image
        """
        # Concatenate all latent inputs
        inputs = [z]
        if c_cat is not None:
            inputs.append(c_cat)
        if c_cont is not None:
            inputs.append(c_cont)
        
        latent = torch.cat(inputs, dim=1)  # (B, latent_dim)
        
        # Project and reshape
        x = self.project(latent)  # (B, ngf*16*4*4)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)  # (B, ngf*16, 4, 4)
        
        # Upsample to target size
        img = self.main(x)
        
        return img


class EMAWrapper:
    """
    Exponential Moving Average wrapper for Generator.
    Maintains shadow weights for stable inference.
    """
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights after each training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights (for evaluation/inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state dict."""
        return self.shadow
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict


def build_generator(config):
    """Build generator from config."""
    model_cfg = config['model']
    
    generator = DCGANGenerator(
        z_dim=model_cfg['z_dim'],
        c_cat_dim=model_cfg['c_cat_dim'],
        c_cont_dim=model_cfg['c_cont_dim'],
        img_size=model_cfg['image_size'],
        base_channels=model_cfg['g']['base_channels'],
        out_channels=model_cfg['g']['out_channels'],
    )
    
    # Initialize weights
    generator.apply(weights_init)
    
    return generator


def weights_init(m):
    """Initialize weights following DCGAN paper but with higher scale."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)  # Increased from 0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)  # Increased from 0.02
        nn.init.constant_(m.bias.data, 0)