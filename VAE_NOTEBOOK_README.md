# MNIST_Generative_Models_VAE.ipynb

## Overview
This notebook trains **only VAE** and uses hardcoded results from a previous 40-epoch training run for GAN, cGAN, and DDPM. This enables faster iteration and comparison while still getting real metrics for VAE.

## Key Features

### 1. VAE Training (Real)
- **Model**: VAE with latent dimension = 20
- **Loss Function**: BCE + KLD (Binary Cross Entropy + Kullback-Leibler Divergence)
  - Changed from MSE to BCE as per assignment requirement
- **Decoder Output**: Sigmoid activation (outputs in [0, 1] range for BCE)
- **Data Preprocessing**: Converts [-1, 1] normalized data to [0, 1] for VAE
- **Epochs**: 40 (matching the hardcoded results for fair comparison)
- **Learning Rate**: 1e-3
- **Real Metrics**: Calculates actual FID, Inception Score, and Training Stability

### 2. Hardcoded Results (GAN, cGAN, DDPM)
Uses pre-computed results from previous 40-epoch training:

**Performance Data:**
- GAN: Clarity=0.600, Stability=0.201, Controllability=0.3
- cGAN: Clarity=0.716, Stability=0.167, Controllability=0.9
- DDPM: Clarity=0.727, Stability=0.974, Controllability=0.8

**Timing Data:**
- GAN: Training=777.2s, Generation=0.001s
- cGAN: Training=771.5s, Generation=0.005s
- DDPM: Training=1710.3s, Generation=3.296s

**Training Curves:**
- Complete 40-epoch loss histories for all models

### 3. Comprehensive Visualizations
All visualizations clearly mark VAE as "TRAINED" and others as "HARDCODED":

1. **Performance Table**: Summary of all metrics and timings
2. **Training Curves**: 2x2 grid showing loss progression
3. **Bar Charts**: Side-by-side comparison of all metrics
4. **Heatmap**: Color-coded performance matrix
5. **Radar Chart**: Multi-dimensional comparison
6. **3D Plot**: Performance space visualization with labels

## Notebook Structure

### Cells Overview
- **Total Cells**: 29 (15 markdown + 14 code)

### Main Sections:
1. Setup and Dependencies
2. Configuration (40 epochs, CALCULATE_REAL_METRICS=True)
3. Data Loading (MNIST dataset)
4. Utility Functions
5. Metrics Calculator (for VAE)
6. VAE Model Implementation (BCE + KLD loss)
7. VAE Training
8. VAE Image Generation
9. VAE Metrics Calculation
10. Hardcoded Results Loading
11. Combine Results
12. Visualization Functions
13. Execute Visualizations
14. Conclusion

## How to Use

### Google Colab (Recommended)
1. Upload `MNIST_Generative_Models_VAE.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all
4. Training takes ~10-15 minutes on GPU
5. Results and visualizations generated automatically

### Local Jupyter
1. Install requirements: `pip install torch torchvision matplotlib seaborn scipy pandas tqdm plotly`
2. Open notebook: `jupyter notebook MNIST_Generative_Models_VAE.ipynb`
3. Run all cells
4. Check `outputs/` folder for saved visualizations

## Key Differences from Original Notebook

| Aspect | Original | VAE-Only Notebook |
|--------|----------|-------------------|
| Training | All 4 models | Only VAE |
| Metrics | All calculated | VAE real, others hardcoded |
| Training Time | ~2-3 hours | ~10-15 minutes |
| Results | All fresh | VAE fresh, others from 40-epoch run |
| Visualizations | All included | All included with labels |

## Output Files

All saved to `outputs/` directory:
- `outputs/checkpoints/vae_epoch_*.pth` - VAE model checkpoints
- `outputs/images/vae/` - Generated VAE images
- `outputs/visualizations/training_curves.png`
- `outputs/visualizations/bar_charts.png`
- `outputs/visualizations/heatmap.png`
- `outputs/visualizations/radar_chart.png`
- `outputs/visualizations/3d_performance.png`

## Assignment Compliance

✅ **VAE Requirements Met:**
- BCE + KLD loss (not MSE)
- Encoder outputs μ and logσ²
- Decoder with Sigmoid activation
- Latent dimension = 20
- Learning rate = 1e-3
- Batch size = 128
- Fixed seed = 42

✅ **Comparison Requirements Met:**
- All four models compared
- Real metrics for VAE
- Consistent 40-epoch comparison
- All visualization types included

## Notes

- **Why Hardcoded Results?** Allows rapid iteration on VAE while maintaining comparison context
- **Fair Comparison**: All models trained for 40 epochs
- **Transparency**: Visualizations clearly label which results are trained vs hardcoded
- **Flexibility**: Easy to retrain other models by swapping hardcoded values with new training runs

## Technical Details

### VAE Loss Function
```python
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

### Data Normalization for VAE
```python
# Convert from [-1, 1] to [0, 1] for BCE loss
data = (data + 1) / 2
```

### Decoder Output Activation
```python
# Sigmoid for [0, 1] output (required for BCE)
nn.Sigmoid()  # instead of nn.Tanh()
```

## Student Information
- **Student**: 7114029008 / 陳鉑琁
- **Assignment**: Comparative Study of VAE, GAN, cGAN, and DDPM
- **Course**: Machine Learning

---

Created: 2025-10-08
Last Updated: 2025-10-08
