# MNIST Generative Models Comparison Project

## Project Overview

This project implements and compares four mainstream generative models on MNIST handwritten digit generation tasks:

1. **Variational Autoencoder (VAE)**
2. **Generative Adversarial Network (GAN)**
3. **Conditional Generative Adversarial Network (cGAN)**
4. **Denoising Diffusion Probabilistic Model (DDPM)**

## 🚀 Quick Start

### Google Colab Execution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/MNIST_COMP/blob/main/MNIST_Generative_Models_Complete.ipynb)

### Local Execution

```bash
git clone https://github.com/Qmo37/MNIST_COMP.git
cd MNIST_COMP
pip install torch torchvision matplotlib numpy tqdm scipy seaborn
# Open MNIST_Generative_Models_Complete.ipynb in Jupyter
```

## 📊 Advanced Visualization and Comparison Methods

### 🎯 Four-Dimensional Evaluation Framework

Our project provides comprehensive evaluation across four key dimensions:

#### 1. **Image Quality (清晰度)**
- **FID Score**: Lower is better (measures similarity to real images)
- **Inception Score (IS)**: Higher is better (quality and diversity)
- **LPIPS Distance**: Perceptual similarity metrics
- **SSIM**: Structural similarity index

#### 2. **Training Stability (穩定性)**
- **Loss Variance**: Standard deviation of training losses
- **Convergence Rate**: Speed of loss stabilization
- **Mode Collapse Detection**: For GAN-based models
- **Training Time Consistency**: Reproducibility across runs

#### 3. **Controllability (可控性)**
- **Conditional Generation**: Ability to generate specific digits
- **Latent Space Interpolation**: Smoothness of transitions
- **Attribute Manipulation**: Control over specific features
- **Sample Diversity**: Range of generated variations

#### 4. **Efficiency (效率)**
- **Training Time**: Time to convergence
- **Inference Speed**: Generation time per sample
- **Memory Usage**: GPU memory requirements
- **Parameter Count**: Model complexity

### 📈 Visualization Methods

#### 1. **Radar Chart** (Best for Overall Comparison)
```python
# All visualizations included in the complete notebook
# MNIST_Generative_Models_Complete.ipynb
```
- **Strengths**: Shows all metrics simultaneously
- **Best for**: Quick model comparison and identifying trade-offs
- **Features**:
  - Normalized scores (0-1 scale)
  - Semi-transparent overlays for comparison
  - Clear legend and performance indicators

#### 2. **3D Spherical Performance Zones** (Innovation)
- **Concept**: Performance as distance from ideal corner (1,1,1)
- **Mathematical Foundation**: Spherical zones based on Euclidean distance
- **Visualization Features**:
  - Semi-transparent spheres for performance levels (0.9, 0.7, 0.5, 0.3)
  - Colored volume inside spheres with low transparency (0.12)
  - Model points clearly positioned in 3D space
  - Golden star marking ideal performance corner

```python
# Performance calculation
distance = sqrt((x-1)² + (y-1)² + (z-1)²)
performance_score = 1 - (distance / sqrt(3))
```

#### 3. **Performance Heatmap**
- **Color Scheme**: Red-Yellow-Green gradient (intuitive bad-to-good mapping)
- **Features**: Numerical values displayed in cells
- **Best for**: Precise metric comparison and identifying specific strengths/weaknesses

### 🔍 Detailed Model Analysis

| Model | Image Quality | Training Stability | Controllability | Efficiency | Overall Score |
|-------|---------------|-------------------|-----------------|------------|---------------|
| **VAE** | 0.70 | 0.90 | 0.60 | 0.80 | 0.750 |
| **GAN** | 0.80 | 0.50 | 0.70 | 0.60 | 0.650 |
| **cGAN** | 0.85 | 0.60 | 0.90 | 0.70 | 0.762 |
| **DDPM** | 0.95 | 0.80 | 0.80 | 0.40 | 0.738 |

#### 3D Spherical Scores (Distance-based)
- **DDPM**: 0.834 (closest to ideal)
- **cGAN**: 0.747
- **VAE**: 0.706
- **GAN**: 0.644 (furthest from ideal)

### 📋 System Requirements

#### Recommended Configuration
- **GPU**: NVIDIA T4 or better (Google Colab free tier)
- **Memory**: At least 8GB RAM
- **Storage**: 2GB available space

#### Software Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (if using GPU)

#### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
tqdm>=4.60.0
scipy>=1.7.0
seaborn>=0.11.0
```

## 🏗️ Project Structure

```
MNIST_COMP/
├── MNIST_Generative_Models_Complete.ipynb  # Complete integrated notebook
├── README.md                               # Project documentation
├── assignment_requirements.txt             # Assignment requirements
├── outputs/                                # Output directory
│   ├── images/                       # Generated images
│   │   ├── vae/
│   │   ├── gan/
│   │   ├── cgan/
│   │   ├── ddpm/
│   │   └── comparison/
│   ├── checkpoints/                  # Model weights
│   ├── metrics/                      # Evaluation metrics
│   └── visualizations/               # Performance charts
│       ├── radar_chart_optimized.png
│       ├── 3d_spherical_zones.png
│       └── performance_heatmap.png
```

## 🎯 Enhanced Features

### 🔥 Latest Improvements

1. **Progress Bars**: Detailed progress tracking for all training processes
2. **Early Stopping**: Prevents overfitting and improves training efficiency
3. **GPU Memory Optimization**: Optimized for T4 GPU performance
4. **Automatic Image Saving**: All generated images saved with timestamps
5. **Advanced Evaluation Metrics**:
   - Comprehensive four-dimensional analysis
   - Spherical performance zone visualization
   - Multi-perspective comparison charts
6. **Interactive Visualizations**: Clear, intuitive performance representations

### 📊 Evaluation Metrics Details

| Metric | Description | Purpose | Range |
|--------|-------------|---------|-------|
| **FID Score** | Fréchet Inception Distance | Similarity to real images | 0-∞ (lower better) |
| **IS Score** | Inception Score | Quality and diversity | 1-∞ (higher better) |
| **Generation Time** | Inference speed | Model practicality | Seconds |
| **Training Stability** | Loss variance | Convergence reliability | 0-1 (higher better) |
| **3D Spherical Score** | Distance from ideal | Overall performance | 0-1 (higher better) |

## 🎨 Comparison Methodologies

### 1. **Quantitative Analysis**
- **Statistical Metrics**: FID, IS, LPIPS, SSIM scores
- **Performance Benchmarking**: Standardized test procedures
- **Reproducibility**: Fixed seeds and controlled environments

### 2. **Qualitative Assessment**
- **Visual Quality**: Human-perceptible image sharpness and realism
- **Diversity**: Range and variety of generated samples
- **Mode Coverage**: Ability to generate all digit classes

### 3. **Efficiency Evaluation**
- **Computational Cost**: Training and inference time measurements
- **Resource Utilization**: Memory and GPU usage profiling
- **Scalability**: Performance under different batch sizes

### 4. **Spherical Performance Zones**
Our innovative 3D visualization approach represents performance as spherical zones:

- **Mathematical Foundation**: Distance from ideal corner (1,1,1)
- **Performance Levels**:
  - Green zone (≥0.9): Excellent performance
  - Yellow zone (≥0.7): Good performance
  - Orange zone (≥0.5): Average performance
  - Red zone (≥0.3): Poor performance
- **Visual Features**:
  - Semi-transparent sphere surfaces
  - Colored volume interiors
  - Model positioning in 3D space

## 🔧 Training Parameters

Standard configuration meeting assignment requirements:

```python
# Fixed parameters
BATCH_SIZE = 128
SEED = 42
EPOCHS = 30  # Adjustable to 50+

# Learning rates
LR_VAE = 1e-3
LR_GAN = 2e-4
LR_DDPM = 1e-3

# Early stopping parameters
PATIENCE = 5
MIN_DELTA = 1e-4
```

## 📈 Detailed Model Comparison

### Performance Matrix

| Model | Clarity | Controllability | Training Efficiency | Inference Efficiency | Stability |
|-------|---------|------------------|---------------------|---------------------|-----------|
| **VAE** | Low (Blurry) | Indirect | 🟢 High | 🟢 Excellent | 🟢 Excellent |
| **GAN** | High (Sharp) | None | 🟡 Medium | 🟢 Excellent | 🔴 Low |
| **cGAN** | High (Sharp) | 🟢 High | 🟡 Medium | 🟢 Excellent | 🔴 Low |
| **DDPM** | 🟢 Excellent | Achievable | 🔴 Very Low | 🔴 Very Low | 🟢 Excellent |

### Trade-off Analysis

1. **Quality vs Speed**: DDPM offers highest quality but slowest generation
2. **Control vs Stability**: cGAN provides control but with training instability
3. **Efficiency vs Performance**: VAE is fastest but with lower image quality
4. **Versatility**: Each model excels in different scenarios

## 🎨 Generated Results

### Output Examples
- **VAE**: 10 random generated images (smooth, slightly blurred)
- **GAN**: 10 random generated images (sharp, potential mode collapse)
- **cGAN**: Digits 0-9, 10 each (100 total, 10×10 grid, controllable)
- **DDPM**: 10 random generated images (highest quality, slow generation)
- **Comparison Chart**: Side-by-side model results with performance metrics

All images automatically saved to `outputs/images/` with timestamps and performance data.

## 🚀 Execution Time Estimates

Expected execution times on Google Colab T4 GPU:

| Model | Training Time | Generation Time | Memory Usage |
|-------|---------------|-----------------|--------------|
| VAE | ~5 minutes | <1 second | 2GB |
| GAN | ~8 minutes | <1 second | 2.5GB |
| cGAN | ~10 minutes | <1 second | 3GB |
| DDPM | ~15 minutes | ~2 minutes | 4GB |

**Total Execution Time**: Approximately 40-45 minutes

## 🔬 Advanced Analysis Features

### 1. **Multi-Perspective Evaluation**
- Radar charts for comprehensive comparison
- 3D spherical zones for intuitive performance understanding
- Heatmaps for detailed metric analysis

### 2. **Performance Insights**
- Best overall performer: **cGAN** (highest average score: 0.762)
- Best in 3D space: **DDPM** (closest to ideal: 0.834)
- Most efficient: **VAE** (best training stability and speed)
- Most controllable: **cGAN** (highest controllability score: 0.90)

### 3. **Visualization Guide**
- **Use Radar Chart**: For quick overall model comparison
- **Use 3D Spherical Zones**: For understanding performance relationships
- **Use Heatmap**: For detailed numerical metric analysis

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Insufficient**
   ```python
   # Memory cleanup included in code
   clear_gpu_memory()
   ```

2. **Training Too Slow**
   ```python
   # Reduce epochs if needed
   EPOCHS = 20
   ```

3. **Colab Disconnection**
   - Models automatically save to checkpoints
   - Can resume training from breakpoints

4. **Visualization Issues**
   ```python
   # Use non-interactive backend for Colab
   import matplotlib
   matplotlib.use('Agg')
   ```

**Note**: Please ensure all dependencies are installed and sufficient GPU memory is available before use. Recommended to run in Google Colab for optimal experience.

## 🎯 Assignment Requirements Checklist

✅ **Fully complies with all assignment requirements**:
- [x] Complete implementation of four models
- [x] MNIST dataset (28×28, grayscale)
- [x] Specified training parameters and loss functions
- [x] Fixed random seed (42)
- [x] All output requirements
- [x] Detailed four-dimensional analysis with advanced visualizations
- [x] Colab environment compatibility
- [x] Comprehensive evaluation framework
- [x] Multi-perspective comparison methods
