# Training Bug Fixes - October 17, 2025

## Issues Encountered and Resolved

### ✅ Issue 1: Cutout Augmentation Index Error
**Error:**
```
TypeError: only integer tensors of a single element can be converted to an index
```

**Root Cause:** 
The `rand_cutout` function was trying to use batched tensors as slice indices, which PyTorch doesn't support.

**Fix Applied:**
Changed from batched indexing to per-image loop with `.item()` conversion:
```python
# Before (broken):
offset_x = torch.randint(..., [x.size(0), 1, 1], ...)
mask[:, :, offset_x:offset_x+cut_w, ...] = 0  # Can't use tensor as index

# After (fixed):
for i in range(x.size(0)):
    offset_x = torch.randint(..., (), ...).item()  # Scalar
    mask[i, :, offset_x:offset_x+cut_w, ...] = 0  # Works!
```

---

### ✅ Issue 2: Translation Augmentation Dimension Reordering
**Error:**
```
RuntimeError: expected input[64, 128, 128, 3] to have 3 channels, but got 128 channels
```

**Root Cause:**
The `rand_translation` function was using advanced indexing that reordered dimensions from `[B, C, H, W]` to `[B, H, W, C]`.

**Fix Applied:**
Rewrote translation to use simple padding and cropping per image:
```python
def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(x.size(0),), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(x.size(0),), device=x.device)
    
    # Pad with maximum shift amount
    x_pad = F.pad(x, [shift_y, shift_y, shift_x, shift_x], mode='constant', value=0)
    
    # Crop per image to apply translation
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        h_start = shift_x + translation_x[i].item()
        w_start = shift_y + translation_y[i].item()
        out[i] = x_pad[i, :, h_start:h_start+H, w_start:w_start+W]
    
    return out
```

---

### ✅ Issue 3: Windows Multiprocessing with DataLoader
**Error:**
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**Root Cause:**
Windows requires special handling for multiprocessing. DataLoader with `num_workers > 0` doesn't work well on Windows without additional guards.

**Fix Applied:**
Set `num_workers: 0` in config file:
```yaml
# configs/dcgan_infogan_128.yaml
data:
  num_workers: 0  # Set to 0 on Windows to avoid multiprocessing issues
```

**Impact:** Training is slightly slower but stable and functional.

---

### ✅ Issue 4: PyTorch AMP Deprecation Warnings
**Warnings:**
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
```

**Root Cause:**
PyTorch 2.0+ deprecated the old `torch.cuda.amp` API in favor of `torch.amp`.

**Fix Applied:**
Updated imports and usage:
```python
# Before:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(enabled=...):

# After:
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')
with autocast('cuda', enabled=...):
```

---

## Final Status

✅ **ALL ISSUES RESOLVED**

Training is now running successfully:
- **Speed**: ~1.5-1.6 iterations/second
- **Batch size**: 64
- **Dataset**: 12,888 images
- **Progress**: Stable, no errors

### Training Metrics
- **Expected time**: ~54 hours for 300,000 steps
- **Current throughput**: ~96-102 images/second
- **GPU**: NVIDIA CUDA device

---

## Files Modified

1. **src/augment/diffaugment.py**
   - Fixed `rand_cutout()` - loop with scalar indices
   - Fixed `rand_translation()` - proper dimension handling

2. **configs/dcgan_infogan_128.yaml**
   - Changed `num_workers: 4` → `num_workers: 0`

3. **src/train.py**
   - Updated `torch.cuda.amp` → `torch.amp`
   - Updated `GradScaler()` → `GradScaler('cuda')`
   - Updated `autocast(enabled=...)` → `autocast('cuda', enabled=...)`

---

## Verification

All fixes verified with test scripts:
- ✅ `test_dataset.py` - Dataset returns correct shape
- ✅ `test_dataloader.py` - DataLoader batches correctly
- ✅ `test_augment.py` - DiffAugment preserves dimensions
- ✅ Training runs without errors

---

## Next Steps

The model is now training successfully. You can:

1. **Monitor progress**: Check `outputs/samples/` for generated images
2. **View checkpoints**: Saved in `outputs/checkpoints/` every 5,000 steps
3. **Track metrics**: D_loss, G_loss, MI_loss logged every 200 steps
4. **Stop/Resume**: Use Ctrl+C to stop, then resume with `--resume` flag

**Estimated completion**: ~54 hours for full 300K steps training

---

Last updated: October 17, 2025
