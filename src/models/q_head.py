"""
Q-head network for InfoGAN.
Predicts latent codes (categorical & continuous) from discriminator features.
Used to compute mutual information loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QHead(nn.Module):
    """
    Auxiliary Q network for InfoGAN.
    Takes discriminator features and predicts latent codes.
    
    Architecture:
        Input: D features (feature_dim, 4, 4)
        Global pooling -> MLP -> [cat_logits, cont_params]
    
    Args:
        feature_dim: Discriminator feature dimension
        c_cat_dim: Categorical code dimension (number of categories)
        c_cont_dim: Continuous code dimension
        hidden_dim: Hidden layer size
    """
    
    def __init__(
        self,
        feature_dim,
        c_cat_dim=8,
        c_cont_dim=3,
        hidden_dim=128,
    ):
        super().__init__()
        
        self.c_cat_dim = c_cat_dim
        self.c_cont_dim = c_cont_dim
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Categorical head (outputs logits)
        if c_cat_dim > 0:
            self.cat_head = nn.Linear(hidden_dim, c_cat_dim)
        
        # Continuous head (outputs mean for Gaussian assumption)
        # In practice, we often assume unit variance for simplicity
        if c_cont_dim > 0:
            self.cont_head = nn.Linear(hidden_dim, c_cont_dim)
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim, 4, 4) from discriminator
        
        Returns:
            cat_logits: (B, c_cat_dim) if c_cat_dim > 0
            cont_params: (B, c_cont_dim) if c_cont_dim > 0
        """
        # Process features
        h = self.shared(features)  # (B, hidden_dim, 4, 4)
        h = F.adaptive_avg_pool2d(h, 1)  # (B, hidden_dim, 1, 1)
        h = h.view(h.size(0), -1)  # (B, hidden_dim)
        
        outputs = {}
        
        # Categorical prediction
        if self.c_cat_dim > 0:
            cat_logits = self.cat_head(h)  # (B, c_cat_dim)
            outputs['cat_logits'] = cat_logits
        
        # Continuous prediction (mean of Gaussian)
        if self.c_cont_dim > 0:
            cont_mean = self.cont_head(h)  # (B, c_cont_dim)
            outputs['cont_mean'] = cont_mean
        
        return outputs


def build_q_head(config, feature_dim):
    """Build Q-head from config."""
    model_cfg = config['model']
    q_cfg = model_cfg['q_head']
    
    q_head = QHead(
        feature_dim=feature_dim,
        c_cat_dim=model_cfg['c_cat_dim'] if q_cfg.get('predict_cat', True) else 0,
        c_cont_dim=model_cfg['c_cont_dim'] if q_cfg.get('predict_cont', True) else 0,
        hidden_dim=q_cfg.get('hidden_dim', 128),
    )
    
    return q_head
