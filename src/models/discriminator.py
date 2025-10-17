"""
Discriminator network for Footballer FaceGAN.
Downsamples 128x128 RGB image to single logit.
Includes spectral normalization for stability.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator with Spectral Normalization.
    
    Architecture:
        Input: (3, 128, 128) RGB image
        Conv blocks: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        Output: (1,) real/fake logit + feature map for Q-head
    
    Args:
        img_size: Input image size (128)
        in_channels: Input channels (3 for RGB)
        base_channels: Base channel multiplier (ndf)
        use_spectral_norm: Apply spectral normalization to conv layers
    """
    
    def __init__(
        self,
        img_size=128,
        in_channels=3,
        base_channels=64,
        use_spectral_norm=True,
    ):
        super().__init__()
        
        ndf = base_channels
        self.use_sn = use_spectral_norm
        
        # Calculate number of downsample blocks
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 (5 blocks)
        assert img_size in [64, 128, 256]
        
        if img_size == 128:
            n_down = 5
        elif img_size == 64:
            n_down = 4
        else:  # 256
            n_down = 6
        
        # Build downsample blocks
        layers = []
        
        # First block: 3 -> ndf, no BN
        layers.append(
            self._make_downsample_block(in_channels, ndf, use_bn=False, use_sn=use_spectral_norm)
        )
        
        # Middle blocks
        in_ch = ndf
        for i in range(1, n_down):
            out_ch = min(in_ch * 2, ndf * 8)
            layers.append(
                self._make_downsample_block(in_ch, out_ch, use_bn=True, use_sn=use_spectral_norm)
            )
            in_ch = out_ch
        
        self.main = nn.Sequential(*layers)
        
        # Final conv to logit
        self.final_conv = nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False)
        if use_spectral_norm:
            self.final_conv = spectral_norm(self.final_conv)
        
        # Store feature dimension for Q-head
        self.feature_dim = in_ch
    
    def _make_downsample_block(self, in_ch, out_ch, use_bn=True, use_sn=True):
        """Create downsample block with Conv2d."""
        layers = []
        
        conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        if use_sn:
            conv = spectral_norm(conv)
        layers.append(conv)
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, img, return_features=False):
        """
        Args:
            img: (B, 3, 128, 128) Input image
            return_features: If True, return intermediate features for Q-head
        
        Returns:
            logit: (B, 1, 1, 1) or (B,) Real/fake logit
            features: (B, feature_dim, 4, 4) if return_features=True
        """
        features = self.main(img)  # (B, feature_dim, 4, 4)
        logit = self.final_conv(features)  # (B, 1, 1, 1)
        
        if return_features:
            return logit.view(-1), features
        else:
            return logit.view(-1)


def build_discriminator(config):
    """Build discriminator from config."""
    model_cfg = config['model']
    
    discriminator = DCGANDiscriminator(
        img_size=model_cfg['image_size'],
        in_channels=model_cfg['d']['in_channels'],
        base_channels=model_cfg['d']['base_channels'],
        use_spectral_norm=model_cfg['d'].get('spectral_norm', True),
    )
    
    # Initialize weights
    discriminator.apply(weights_init)
    
    return discriminator


def weights_init(m):
    """Initialize weights following DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
