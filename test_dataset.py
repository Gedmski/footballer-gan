import sys
sys.path.insert(0, 'src')
from datasets.fm_cutout import FM23CutoutDataset
import yaml

config = yaml.safe_load(open('configs/dcgan_infogan_128.yaml'))
ds = FM23CutoutDataset(
    'data/processed', 
    128, 
    True, 
    (128,128,128), 
    True, 
    config['data']['normalize']
)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Sample type: {type(sample)}')
print(f'Sample shape: {sample.shape}')
print(f'Sample dtype: {sample.dtype}')
print(f'Sample min/max: {sample.min()}/{sample.max()}')
