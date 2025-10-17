"""
Data preprocessing script for FM23 Cutout Facepack.
Converts raw images to processed format (128x128, RGB, center-cropped).
"""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def preprocess_image(img_path, output_path, size=128, bg_color=(128, 128, 128)):
    """
    Preprocess a single image.
    
    Args:
        img_path: Path to input image
        output_path: Path to output image
        size: Target size (square)
        bg_color: Background color for RGBA conversion
    """
    try:
        img = Image.open(img_path)
        
        # Convert RGBA to RGB
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, bg_color)
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Center crop to square
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize
        img = img.resize((size, size), Image.LANCZOS)
        
        # Save
        img.save(output_path, quality=95)
        
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def preprocess_dataset(raw_dir, processed_dir, size=128, bg_color=(128, 128, 128)):
    """
    Preprocess all images in raw directory.
    
    Args:
        raw_dir: Path to raw images directory
        processed_dir: Path to output directory
        size: Target size
        bg_color: Background color for RGBA
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(raw_dir.glob(f'**/*{ext}'))
        image_paths.extend(raw_dir.glob(f'**/*{ext.upper()}'))
    
    print(f"Found {len(image_paths)} images in {raw_dir}")
    
    # Process each image
    success_count = 0
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Create output filename
        output_filename = img_path.stem + '.png'
        output_path = processed_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            success_count += 1
            continue
        
        # Process
        if preprocess_image(img_path, output_path, size, bg_color):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(image_paths)} images")
    print(f"Output directory: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess FM23 Cutout Facepack')
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Path to raw images directory'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='data/processed',
        help='Path to output directory'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=128,
        help='Target image size (square)'
    )
    parser.add_argument(
        '--bg_color',
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help='Background color for RGBA (R G B)'
    )
    args = parser.parse_args()
    
    preprocess_dataset(
        args.raw_dir,
        args.processed_dir,
        args.size,
        tuple(args.bg_color)
    )


if __name__ == "__main__":
    main()
