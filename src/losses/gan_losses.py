"""
GAN loss functions.
Implements non-saturating, LSGAN, WGAN-GP variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


def nonsaturating_loss_dis(real_logits, fake_logits, label_smoothing=0.0):
    """
    Non-saturating GAN loss for discriminator.
    D_loss = -E[log D(x)] - E[log(1 - D(G(z)))]
    
    Args:
        real_logits: Discriminator output on real images
        fake_logits: Discriminator output on fake images
        label_smoothing: One-sided label smoothing (0.0 to 0.2)
    """
    real_target = 1.0 - label_smoothing
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits) * real_target
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return real_loss + fake_loss


def nonsaturating_loss_gen(fake_logits):
    """
    Non-saturating GAN loss for generator.
    G_loss = -E[log D(G(z))]
    """
    return F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )


def lsgan_loss_dis(real_logits, fake_logits):
    """
    Least-Squares GAN loss for discriminator.
    D_loss = 0.5 * E[(D(x) - 1)^2] + 0.5 * E[D(G(z))^2]
    """
    real_loss = 0.5 * torch.mean((real_logits - 1) ** 2)
    fake_loss = 0.5 * torch.mean(fake_logits ** 2)
    return real_loss + fake_loss


def lsgan_loss_gen(fake_logits):
    """
    Least-Squares GAN loss for generator.
    G_loss = 0.5 * E[(D(G(z)) - 1)^2]
    """
    return 0.5 * torch.mean((fake_logits - 1) ** 2)


def wgan_loss_dis(real_logits, fake_logits):
    """
    Wasserstein GAN loss for discriminator (without GP).
    D_loss = -E[D(x)] + E[D(G(z))]
    """
    return -torch.mean(real_logits) + torch.mean(fake_logits)


def wgan_loss_gen(fake_logits):
    """
    Wasserstein GAN loss for generator.
    G_loss = -E[D(G(z))]
    """
    return -torch.mean(fake_logits)


def gradient_penalty(discriminator, real_images, fake_images, device):
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_images: Real images batch
        fake_images: Generated images batch
        device: torch device
    
    Returns:
        gp: Gradient penalty scalar
    """
    batch_size = real_images.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient penalty
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gp


def r1_regularization(discriminator, real_images):
    """
    R1 regularization (gradient penalty on real data only).
    Used in StyleGAN2.
    
    Args:
        discriminator: Discriminator network
        real_images: Real images batch (requires grad)
    
    Returns:
        r1_penalty: R1 penalty scalar
    """
    real_images = real_images.requires_grad_(True)
    real_logits = discriminator(real_images)
    
    # Compute gradients w.r.t. real images
    gradients = autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # R1 penalty: ||∇D(x)||^2
    r1_penalty = gradients.pow(2).view(gradients.size(0), -1).sum(1).mean()
    
    return r1_penalty


class GANLoss:
    """
    Unified GAN loss wrapper.
    """
    
    def __init__(self, loss_type='nonsat', label_smoothing=0.0, gp_lambda=10.0, device='cuda'):
        """
        Args:
            loss_type: 'nonsat', 'lsgan', or 'wgan_gp'
            label_smoothing: Label smoothing for nonsat loss
            gp_lambda: Gradient penalty weight for WGAN-GP
            device: torch device
        """
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.gp_lambda = gp_lambda
        self.device = device
    
    def dis_loss(self, real_logits, fake_logits, real_images=None, fake_images=None, discriminator=None):
        """Compute discriminator loss."""
        if self.loss_type == 'nonsat':
            loss = nonsaturating_loss_dis(real_logits, fake_logits, self.label_smoothing)
        elif self.loss_type == 'lsgan':
            loss = lsgan_loss_dis(real_logits, fake_logits)
        elif self.loss_type == 'wgan_gp':
            loss = wgan_loss_dis(real_logits, fake_logits)
            if self.gp_lambda > 0 and discriminator is not None:
                gp = gradient_penalty(discriminator, real_images, fake_images, self.device)
                loss = loss + self.gp_lambda * gp
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def gen_loss(self, fake_logits):
        """Compute generator loss."""
        if self.loss_type == 'nonsat':
            loss = nonsaturating_loss_gen(fake_logits)
        elif self.loss_type == 'lsgan':
            loss = lsgan_loss_gen(fake_logits)
        elif self.loss_type == 'wgan_gp':
            loss = wgan_loss_gen(fake_logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


def build_gan_loss(config):
    """Build GAN loss from config."""
    loss_cfg = config['loss']
    
    loss_type = loss_cfg.get('gan_loss', 'nonsat')
    
    gan_loss = GANLoss(
        loss_type=loss_type,
        label_smoothing=loss_cfg.get('d_label_smoothing', 0.0),
        gp_lambda=loss_cfg.get('gp_lambda', 10.0),
        device=config.get('device', 'cuda'),
    )
    
    return gan_loss
