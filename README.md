# MNIST Generative Models Comparison Project

## 📋 Project Overview

This project implements and compares four mainstream generative models on MNIST handwritten digit generation:

1. **VAE** (Variational Autoencoder)
2. **GAN** (Generative Adversarial Network)
3. **cGAN** (Conditional GAN)
4. **DDPM** (Denoising Diffusion Probabilistic Model)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/MNIST_COMP/blob/main/MNIST_Generative_Models_Complete.ipynb)

### Option 2: Local Jupyter Notebook
```bash
git clone https://github.com/Qmo37/MNIST_COMP.git
cd MNIST_COMP
pip install -r requirements.txt
jupyter notebook MNIST_Generative_Models_Complete.ipynb
```

### Option 3: Python Script
```bash
python mnist_generative_models.py --epochs 30 --calculate-real-metrics
```

## 📁 Project Structure

```
MNIST_COMP/
├── 📄 Main Files
│   ├── MNIST_Generative_Models_Complete.ipynb    # 🎯 Main notebook (START HERE)
│   ├── mnist_generative_models.py                # Alternative Python script
│   ├── README.md                                  # This file
│   ├── assignment_requirements.txt                # Assignment specifications
│   └── HW2-- 大亂鬥.pdf                           # Assignment document
│
├── 📁 src/                                        # Source utilities
│   ├── visualization_functions.py                 # Reusable visualization code
│   └── performance_plot.py                        # Performance plotting utilities
│
├── 📁 scripts/                                    # Helper scripts
│   ├── migration/                                 # Notebook migration tools
│   │   ├── complete_migration.py
│   │   ├── MIGRATION_SUMMARY.md
│   │   └── QUICK_REFERENCE.md
│   └── visualizations/                            # Visualization experiments
│       ├── interactive_3d_visualization.py
│       ├── colab_interactive_3d.py
│       └── performance_zones_3d.py
│
├── 📁 outputs/                                    # Generated results
│   ├── images/                                    # Model outputs
│   │   ├── vae/
│   │   ├── gan/
│   │   ├── cgan/
│   │   ├── ddpm/
│   │   └── comparison/
│   ├── visualizations/                            # Performance charts
│   │   ├── radar_chart_optimized.png
│   │   ├── 3d_spherical_zones.png
│   │   ├── performance_heatmap.png
│   │   └── 3d_spherical_performance.png
│   └── checkpoints/                               # Model weights
│
├── 📁 data/                                       # Dataset (auto-downloaded)
│   └── MNIST/
│
└── 📁 backups/                                    # Version backups
    ├── MNIST_Generative_Models_Complete.ipynb.fixed
    └── MNIST_Generative_Models_Complete.ipynb.original
```

## 🎯 Four-Dimensional Evaluation Framework

### 1. **Image Quality (清晰度)**
- FID Score (lower is better)
- Inception Score (higher is better)
- Visual sharpness and realism

### 2. **Training Stability (穩定性)**
- Loss variance
- Convergence rate
- Mode collapse detection

### 3. **Controllability (可控性)**
- Conditional generation ability
- Latent space manipulation
- Sample diversity

### 4. **Efficiency (效率)**
- Training time
- Inference speed
- Memory usage

## 📊 Model Comparison Results

| Model | Image Quality | Stability | Controllability | Efficiency | Overall |
|-------|--------------|-----------|-----------------|------------|---------|
| **VAE** | 0.70 | 0.90 | 0.60 | 0.80 | 0.750 |
| **GAN** | 0.80 | 0.50 | 0.70 | 0.60 | 0.650 |
| **cGAN** | 0.85 | 0.60 | 0.90 | 0.70 | 0.762 |
| **DDPM** | 0.95 | 0.80 | 0.80 | 0.40 | 0.738 |

### Key Findings
- 🏆 **Best Quality**: DDPM (0.95)
- 🏆 **Best Control**: cGAN (0.90)
- 🏆 **Best Stability**: VAE (0.90)
- 🏆 **Best Efficiency**: VAE (0.80)
- 🏆 **Best Overall**: cGAN (0.762)

## 📈 Visualization Methods

### 1. Radar Chart
Perfect for quick overall comparison across all metrics.

### 2. 3D Spherical Performance Zones
Innovative visualization showing performance as distance from ideal point (1,1,1).
- Smaller sphere = Better performance
- Color-coded zones for different performance levels

### 3. Performance Heatmap
Detailed numerical comparison with color gradients.

### 4. Training Curves
Loss progression over epochs for each model.

## 🔧 Training Parameters

Assignment-compliant configuration:

```python
BATCH_SIZE = 128        # Assignment requirement
LATENT_DIM = 100        # GAN latent dimension
SEED = 42               # Fixed seed for reproducibility
EPOCHS = 30             # Adjustable (5 for quick test)

# Learning rates (Assignment requirements)
LR_VAE = 1e-3           # VAE learning rate
LR_GAN = 2e-4           # GAN/cGAN learning rate
LR_DDPM = 1e-3          # DDPM learning rate
```

## ⚡ Performance Estimates (Google Colab T4)

| Model | Training Time | Generation Time | Memory |
|-------|--------------|-----------------|--------|
| VAE | ~5 min | <1 sec | 2 GB |
| GAN | ~8 min | <1 sec | 2.5 GB |
| cGAN | ~10 min | <1 sec | 3 GB |
| DDPM | ~15 min | ~2 min | 4 GB |

**Total Runtime**: ~40-45 minutes

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
tqdm>=4.60.0
scipy>=1.7.0
seaborn>=0.11.0
pandas>=1.3.0
plotly>=5.0.0  # Optional for interactive 3D
```

## 🎨 Generated Outputs

### Image Outputs
- **VAE**: 10 random images (smooth, slightly blurred)
- **GAN**: 10 random images (sharp, potential mode collapse)
- **cGAN**: 10×10 grid (digits 0-9, 10 each, controllable)
- **DDPM**: 10 random images (highest quality, slow generation)
- **Comparison**: Side-by-side comparison figure

### Visualization Outputs
- Radar chart comparison
- 3D spherical performance zones
- Performance heatmap
- Training loss curves
- Metrics summary table

All outputs saved to `outputs/` directory.

## 🐛 Troubleshooting

### GPU Memory Issues
```python
# Memory cleanup is built-in
clear_gpu_memory()
```

### Training Too Slow
```python
# Reduce epochs for quick testing
EPOCHS = 5
CALCULATE_REAL_METRICS = False
```

### Colab Disconnection
- Checkpoints auto-saved every 10 epochs
- Resume from checkpoint if needed

### Visualization Issues
```python
# For headless environments
import matplotlib
matplotlib.use('Agg')
```

## ✅ Assignment Compliance Checklist

- [x] Four models implemented (VAE, GAN, cGAN, DDPM)
- [x] MNIST dataset (28×28 grayscale)
- [x] Batch size: 128
- [x] Fixed seed: 42
- [x] Correct learning rates (1e-3 VAE, 2e-4 GAN/cGAN)
- [x] Label smoothing for cGAN (0.9)
- [x] BCE + KLD loss (VAE)
- [x] BCE adversarial loss (GAN/cGAN)
- [x] MSE denoising loss (DDPM)
- [x] All required outputs generated
- [x] Four-dimensional analysis
- [x] Comprehensive visualizations
- [x] Colab compatible

## 📚 Documentation

- **MIGRATION_SUMMARY.md**: Details about notebook migration from Python script
- **QUICK_REFERENCE.md**: Quick reference guide for common tasks
- **assignment_requirements.txt**: Original assignment specifications

## 🔗 Links

- [GitHub Repository](https://github.com/Qmo37/MNIST_COMP)
- [Google Colab Notebook](https://colab.research.google.com/github/Qmo37/MNIST_COMP/blob/main/MNIST_Generative_Models_Complete.ipynb)

## 📄 License

This project is for educational purposes as part of a machine learning course assignment.

---

**Author**: Qmo37  
**Last Updated**: October 2024  
**Course**: Machine Learning - Generative Models Assignment
