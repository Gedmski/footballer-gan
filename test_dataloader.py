import sys
sys.path.insert(0, 'src')
from datasets.fm_cutout import get_fm_dataloader
import yaml

if __name__ == '__main__':
    config = yaml.safe_load(open('configs/dcgan_infogan_128.yaml'))
    # Test with 0 workers first
    config['data']['num_workers'] = 0
    loader = get_fm_dataloader(config, is_train=True)
    print(f'DataLoader created')

    for batch in loader:
        print(f'Batch type: {type(batch)}')
        print(f'Batch shape: {batch.shape}')
        print(f'Batch dtype: {batch.dtype}')
        break
