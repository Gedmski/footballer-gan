"""
FM23 Cutout Facepack Dataset Loader.
Loads footballer face images from raw/processed directories.
Handles RGBA->RGB conversion and normalization.
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FM23CutoutDataset(Dataset):
    """
    Custom dataset for Football Manager cutout faces.
    
    Args:
        data_dir: Path to processed images directory
        img_size: Target image size (assumes square)
        rgba_to_rgb: Convert RGBA to RGB
        rgba_bg_color: Background color for RGBA conversion [R, G, B]
        center_crop: Apply center crop before resize
        normalize: Dict with 'mean' and 'std' for normalization
    """
    
    def __init__(
        self,
        data_dir,
        img_size=128,
        rgba_to_rgb=True,
        rgba_bg_color=(128, 128, 128),
        center_crop=True,
        normalize=None,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.rgba_to_rgb = rgba_to_rgb
        self.rgba_bg_color = rgba_bg_color
        
        # Find all valid image files
        self.image_paths = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        
        for ext in valid_extensions:
            self.image_paths.extend(self.data_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.data_dir.glob(f'*{ext.upper()}'))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Build transforms
        transform_list = []
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(min(img_size, img_size)))
        
        transform_list.append(transforms.Resize((img_size, img_size), antialias=True))
        transform_list.append(transforms.ToTensor())
        
        if normalize:
            mean = normalize.get('mean', [0.5, 0.5, 0.5])
            std = normalize.get('std', [0.5, 0.5, 0.5])
            transform_list.append(transforms.Normalize(mean, std))
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(img_path)
        
        # Handle RGBA conversion
        if self.rgba_to_rgb and img.mode == 'RGBA':
            # Create RGB background
            bg = Image.new('RGB', img.size, self.rgba_bg_color)
            bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms
        img = self.transform(img)
        
        return img


def get_fm_dataloader(config, is_train=True):
    """
    Create DataLoader for FM23 dataset from config.
    
    Args:
        config: Configuration dict
        is_train: If True, use training settings
    
    Returns:
        DataLoader instance
    """
    data_cfg = config['data']
    paths_cfg = config['paths']
    
    # Use processed data directory
    data_dir = paths_cfg['data_processed']
    
    # Create dataset
    dataset = FM23CutoutDataset(
        data_dir=data_dir,
        img_size=data_cfg['img_size'],
        rgba_to_rgb=data_cfg.get('rgba_to_rgb', True),
        rgba_bg_color=tuple(data_cfg.get('rgba_bg_color', [128, 128, 128])),
        center_crop=data_cfg.get('center_crop', True),
        normalize=data_cfg.get('normalize'),
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=data_cfg.get('shuffle', True) if is_train else False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        drop_last=is_train,
    )
    
    return dataloader
