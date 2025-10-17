import torch
import sys
sys.path.insert(0, 'src')
from augment.diffaugment import DiffAugment

# Test DiffAugment doesn't mess up dimensions
x = torch.randn(4, 3, 128, 128)  # [B, C, H, W]
print(f"Input shape: {x.shape}")

# Test each augmentation
x_color = DiffAugment(x, policy='color')
print(f"After color: {x_color.shape}")

x_trans = DiffAugment(x, policy='translation')
print(f"After translation: {x_trans.shape}")

x_cutout = DiffAugment(x, policy='cutout')
print(f"After cutout: {x_cutout.shape}")

x_all = DiffAugment(x, policy='color,translation,cutout')
print(f"After all: {x_all.shape}")

assert x_all.shape == torch.Size([4, 3, 128, 128]), "Shape mismatch!"
print("\n✓ All augmentations preserve shape correctly!")
