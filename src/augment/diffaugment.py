# src/augment/diffaugment.py
# Reference: Zhao et al. 2020 (DiffAugment: Data Augmentation for GANs)
import torch
import torch.nn.functional as F

def DiffAugment(x, policy='color,translation,cutout', channels_first=True):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
    return x.contiguous()

def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + x_mean

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + x_mean

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(x.size(0),), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(x.size(0),), device=x.device)
    
    # Pad the image with maximum shift on all sides
    x_pad = F.pad(x, [shift_y, shift_y, shift_x, shift_x], mode='constant', value=0)
    
    # Apply translation by cropping from the padded image
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        h_start = shift_x + translation_x[i].item()
        w_start = shift_y + translation_y[i].item()
        out[i] = x_pad[i, :, h_start:h_start+H, w_start:w_start+W]
    
    return out

def rand_cutout(x, ratio=0.5):
    cut_w, cut_h = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    mask = torch.ones_like(x)
    for i in range(x.size(0)):
        offset_x = torch.randint(0, x.size(2) - cut_w + 1, (), device=x.device).item()
        offset_y = torch.randint(0, x.size(3) - cut_h + 1, (), device=x.device).item()
        mask[i, :, offset_x:offset_x+cut_w, offset_y:offset_y+cut_h] = 0
    return x * mask

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
