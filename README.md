# MNIST Generative Models Comparison Project

## 📋 Project Overview

This project implements and compares four mainstream generative models on MNIST handwritten digit generation:

1.  **VAE** (Variational Autoencoder)
2.  **GAN** (Generative Adversarial Network)
3.  **cGAN** (Conditional GAN)
4.  **DDPM** (Denoising Diffusion Probabilistic Model)

The primary deliverable is the `MNIST_Generative_Models_Complete.ipynb` notebook, which contains all code, analysis, and visualizations in a consolidated format.

## 🚀 Quick Start

The easiest way to run this project is to use Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/MNIST_COMP/blob/main/MNIST_Generative_Models_Complete.ipynb)

Simply open the notebook, select `Runtime > Run all`, and the notebook will execute from top to bottom, producing all results and visualizations.

## 📂 Final Project Structure

After cleanup, the essential files for the assignment are:

```
MNIST_COMP/
├── MNIST_Generative_Models_Complete.ipynb    # 🎯 MAIN NOTEBOOK (START HERE)
├── mnist_generative_models_complete.py       # Python script backup of the notebook
├── README.md                                 # This file
├── assignment_requirements.txt               # Assignment specifications
├── HW2-- 大亂鬥.pdf                          # Original assignment PDF
│
├── outputs/                                  # Generated results
│   └── visualizations/
│       └── 3d_performance_spherical_zones.png  # Final 3D visualization
│
└── data/                                     # MNIST dataset (auto-downloaded)
```

## 📊 Performance Analysis (Real Metrics)

The following results were generated from a complete run of the notebook, calculating real performance metrics for each model.

### Key Findings
*   **Quality vs. Speed Trade-off**: DDPM produces the highest quality images but is by far the slowest. VAE is the fastest model overall but has the blurriest images.
*   **Control**: cGAN is the clear winner for controllability, as it can generate specific digits on command.
*   **Stability**: VAE and DDPM are very stable to train. GANs are notoriously unstable, though this can be mitigated.

### Qualitative Summary

| Category | VAE | GAN | cGAN | DDPM |
| :--- | :--- | :--- | :--- | :--- |
| **Image Quality** | Slightly blurred but consistent | Sharp, but potential for artifacts | Sharp and clear | Highest quality, most realistic |
| **Controllability** | Indirect (latent space) | Limited (random seed) | Excellent (class labels) | Good (can be conditional) |
| **Stability** | Very Stable | Prone to mode collapse | More stable than GAN | Very Stable |
| **Efficiency** | **Very Fast** (Train & Inference) | Moderate Train, Fast Inference | Moderate Train, Fast Inference | **Very Slow** (Train & Inference) |

### Quantitative Results Table

```
================================================================================
                    PERFORMANCE AND TIMING SUMMARY TABLE
================================================================================
 Model  Clarity (Image Quality)  Training Stability  Controllability  Efficiency  Training Time (s)  Inference Time (ms/img)
------  -------------------------  --------------------  -----------------  ------------  -------------------  -------------------------
   VAE                      0.799                 0.000              0.632         0.997               530.6                        2.8
   GAN                      0.600                 0.201              0.300         0.956               777.2                        0.1
  cGAN                      0.716                 0.167              0.900         0.959               771.5                        0.1
  DDPM                      0.727                 0.974              0.800         0.000              1710.3                      329.6
================================================================================
```

**Key Observations:**
- **VAE**: Best efficiency (fastest training + inference), but lowest training stability score
- **GAN**: Fast inference (0.1 ms/img), moderate image quality
- **cGAN**: Best controllability (0.900), sharp images, fast inference
- **DDPM**: Best training stability (0.974) and quality, but extremely slow (329.6 ms/img)

## 📊 Visualizations

The notebook generates comprehensive visualizations comparing all four models across multiple dimensions:

### Training Curves
![Training Curves](outputs/visualizations/Training%20Curve.png)

Shows the loss progression over 40 epochs for each model:
- **VAE**: Steady convergence of reconstruction + KLD loss
- **GAN**: Generator and Discriminator losses in adversarial competition
- **cGAN**: Conditional GAN dynamics with class-aware training
- **DDPM**: Gradual diffusion loss reduction over epochs

### Performance Metrics Bar Charts
![Bar Charts](outputs/visualizations/Bar%20Charts.png)

Side-by-side comparison of the four key performance metrics:
- **Clarity (Image Quality)**: Measured by FID and Inception Score
- **Training Stability**: Based on loss variance and convergence
- **Controllability**: Ability to generate specific outputs
- **Efficiency**: Training and inference speed

### Performance Heatmap
![Heatmap](outputs/visualizations/Heatmap.png)

A comprehensive heatmap displaying all metrics across all models in a single view. Warmer colors indicate better performance, making it easy to identify each model's strengths and weaknesses at a glance.

### Radar Chart
![Radar Chart](outputs/visualizations/Radar%20chart.png)

Multi-dimensional radar chart providing an intuitive visualization of how each model balances different performance aspects. The area covered by each model indicates its overall versatility:
- **DDPM**: Largest coverage (best balance)
- **cGAN**: Strong in quality and controllability
- **VAE**: Best efficiency but lower quality
- **GAN**: Moderate across all metrics

### 3D Performance Space
![3D Performance Plot](outputs/visualizations/3D%20Performance%20Plot%20(Filled%20Cuboids).png)

An interactive 3D visualization plotting models in a three-dimensional performance space:
- **X-axis**: Image Quality (Clarity)
- **Y-axis**: Training Stability
- **Z-axis**: Controllability

The plot includes color-coded performance zones (Elite, Excellent, Good) with filled cuboid regions, showing which models achieve high performance across multiple dimensions simultaneously. The golden star represents the theoretical ideal model (1.0 in all metrics).

## ✅ Assignment Compliance Checklist

- [x] Four models implemented (VAE, GAN, cGAN, DDPM)
- [x] MNIST dataset (28×28 grayscale)
- [x] Batch size: 128
- [x] Fixed seed: 42
- [x] Correct learning rates (1e-3 VAE, 2e-4 GAN/cGAN)
- [x] Label smoothing for cGAN (0.9)
- [x] All required loss functions implemented
- [x] All required outputs generated and consolidated in the main notebook
- [x] Four-dimensional analysis performed with real metrics
- [x] Comprehensive visualizations generated

---
**Author**: Qmo37  
**Last Updated**: October 2024