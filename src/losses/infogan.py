"""
InfoGAN mutual information loss.
Maximizes mutual information between latent codes and generated images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def infogan_loss(q_outputs, c_cat_target=None, c_cont_target=None):
    """
    Compute InfoGAN mutual information loss.
    
    The loss encourages the generator to use the latent codes meaningfully
    by maximizing I(c; G(z, c)), which is approximated by minimizing:
        -E[log Q(c | G(z, c))]
    
    For categorical codes: use cross-entropy
    For continuous codes: use Gaussian negative log-likelihood (MSE)
    
    Args:
        q_outputs: Dict with 'cat_logits' and/or 'cont_mean'
        c_cat_target: (B, c_cat_dim) Target categorical codes (one-hot or indices)
        c_cont_target: (B, c_cont_dim) Target continuous codes
    
    Returns:
        loss: Mutual information loss (lower is better)
        stats: Dict with loss components
    """
    loss = 0.0
    stats = {}
    
    # Categorical mutual information
    if 'cat_logits' in q_outputs and c_cat_target is not None:
        cat_logits = q_outputs['cat_logits']  # (B, c_cat_dim)
        
        # If c_cat_target is one-hot, convert to indices
        if c_cat_target.dim() > 1 and c_cat_target.size(1) > 1:
            c_cat_target = c_cat_target.argmax(dim=1)  # (B,)
        
        # Cross-entropy loss
        cat_loss = F.cross_entropy(cat_logits, c_cat_target.long())
        loss = loss + cat_loss
        stats['mi_cat'] = cat_loss.item()
    
    # Continuous mutual information
    # Assume Q predicts mean of Gaussian with unit variance
    # NLL = 0.5 * ||c - Q(c|x)||^2 + const
    if 'cont_mean' in q_outputs and c_cont_target is not None:
        cont_mean = q_outputs['cont_mean']  # (B, c_cont_dim)
        
        # MSE loss (equivalent to Gaussian NLL with unit variance)
        cont_loss = F.mse_loss(cont_mean, c_cont_target)
        loss = loss + cont_loss
        stats['mi_cont'] = cont_loss.item()
    
    return loss, stats


class InfoGANLoss:
    """
    InfoGAN loss wrapper.
    """
    
    def __init__(self, mi_weight=1.0):
        """
        Args:
            mi_weight: Weight for mutual information loss
        """
        self.mi_weight = mi_weight
    
    def __call__(self, q_outputs, c_cat=None, c_cont=None):
        """
        Compute weighted InfoGAN loss.
        
        Returns:
            loss: Weighted MI loss
            stats: Loss statistics
        """
        mi_loss, stats = infogan_loss(q_outputs, c_cat, c_cont)
        loss = self.mi_weight * mi_loss
        stats['mi_total'] = mi_loss.item()
        stats['mi_weighted'] = loss.item()
        return loss, stats


def sample_categorical(batch_size, n_categories, device='cuda'):
    """
    Sample categorical codes uniformly.
    
    Args:
        batch_size: Batch size
        n_categories: Number of categories
        device: torch device
    
    Returns:
        c_cat: (batch_size, n_categories) One-hot encoded
    """
    indices = torch.randint(0, n_categories, (batch_size,), device=device)
    c_cat = F.one_hot(indices, num_classes=n_categories).float()
    return c_cat


def sample_continuous(batch_size, n_cont, dist='uniform', device='cuda'):
    """
    Sample continuous codes.
    
    Args:
        batch_size: Batch size
        n_cont: Number of continuous dimensions
        dist: 'uniform' (-1, 1) or 'normal' (0, 1)
        device: torch device
    
    Returns:
        c_cont: (batch_size, n_cont)
    """
    if dist == 'uniform':
        c_cont = torch.rand(batch_size, n_cont, device=device) * 2 - 1  # [-1, 1]
    elif dist == 'normal':
        c_cont = torch.randn(batch_size, n_cont, device=device)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    
    return c_cont


def build_infogan_loss(config):
    """Build InfoGAN loss from config."""
    loss_cfg = config['loss']
    
    infogan_loss = InfoGANLoss(
        mi_weight=loss_cfg.get('mi_weight', 1.0)
    )
    
    return infogan_loss
