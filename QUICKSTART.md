# Quick Start Guide - Footballer FaceGAN

## 🚀 Setup (5 minutes)

### 1. Create Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your FM23 cutout face images in `data/raw/`, then run:

```powershell
python src/preprocess_data.py --raw_dir data/raw --processed_dir data/processed --size 128
```

**Expected output**: Cropped 128×128 RGB PNG files in `data/processed/`

---

## 🏋️ Training (12 hours on RTX 4060)

```powershell
python src/train.py --config configs/dcgan_infogan_128.yaml --device cuda
```

**What happens:**
- Checkpoints saved every 5,000 steps in `outputs/checkpoints/`
- Sample images every 1,000 steps in `outputs/samples/`
- EMA generator saved as `outputs/checkpoints/ema_latest.pt`

**Monitor progress:**
- Check `outputs/samples/` for visual quality
- Training logs show D_loss, G_loss, MI_loss

**Typical convergence:**
- 50K steps: Blurry faces, visible structure
- 100K steps: Recognizable faces, some artifacts
- 200K steps: Sharp faces, good diversity
- 300K steps: High quality, stable

---

## 🎨 Interactive Demo

After training (or download pretrained checkpoint):

```powershell
python src/app/gradio_app.py --checkpoint outputs/checkpoints/ema_latest.pt
```

Open browser to `http://localhost:7860`

**Controls:**
- **Seed**: Reproducible randomness
- **Category (0-7)**: Discrete face clusters
- **c_cont[0-2]**: Lighting, complexion, face shape
- **Truncation ψ**: Quality vs. diversity (0.7 = safer, 1.5 = wilder)

---

## 📊 Evaluation

Compute FID/KID metrics:

```powershell
python src/eval_fid_kid.py --checkpoint outputs/checkpoints/ema_latest.pt --real data/processed --num_gen 5000
```

**Targets:**
- FID < 30 = Good
- KID < 0.05 = Good

---

## 🔍 Latent Analysis

Visualize PCA components:

```powershell
python src/viz/latent_pca.py --checkpoint outputs/checkpoints/ema_latest.pt --num_samples 10000 --output_dir outputs/pca_analysis
```

Output: Variance plots and 2D projections in `outputs/pca_analysis/`

---

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in config (64 → 32 → 16)
- Disable AMP: `amp: false` in config
- Use gradient accumulation: `grad_accum_steps: 2`

### Mode Collapse
- Check DiffAugment is enabled: `augment.enabled: true`
- Increase MI weight: `loss.mi_weight: 1.5`
- Lower D learning rate: `optim.d.lr: 0.00005`

### Slow Training
- Enable cudnn benchmark: `cudnn_benchmark: true` (default)
- Reduce `num_workers` if CPU-bound: `data.num_workers: 2`
- Use smaller FID evaluation: `eval.num_gen_for_fid: 2000`

### No GPU
Set `device: cpu` in config (very slow, 100× slower)

---

## 📝 Tips

1. **Start small**: Train 10K steps first to verify setup
2. **Save checkpoints**: Don't rely only on latest (corruption risk)
3. **Visual inspection**: FID is useful but check samples yourself
4. **Seed control**: Use same seed for reproducible comparisons
5. **Data quality**: Remove low-quality/corrupted images from dataset

---

## 🎯 Next Steps

After successful training:

1. **Generate samples**: Use Gradio app to explore latent space
2. **Analyze codes**: Run PCA to understand learned features
3. **Fine-tune**: Adjust continuous codes to discover attributes
4. **Export**: Save favorite generations for presentation
5. **Scale up**: Try 256×256 resolution or larger dataset

---

## 📚 Documentation

- Full paper: `reports/paper.md`
- Config reference: `configs/dcgan_infogan_128.yaml`
- Architecture details: `src/models/`

---

## 🆘 Support

Issues? Check:
1. Python version: 3.10+ recommended
2. CUDA compatibility: Match PyTorch version
3. Disk space: ~10GB for checkpoints + samples
4. Dataset size: At least 1,000 images for stable training

---

Happy generating! ⚽🎨
