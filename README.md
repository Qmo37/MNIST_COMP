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
                          ASSIGNMENT COMPARISON TABLE
================================================================================
 Model   Clarity (Image Quality)   Training Stability   Controllability   Efficiency  Training Time (s)  Generation Time (s)  Average Score
------  -------------------------  --------------------  -----------------  ------------  -------------------  -------------------  ---------------
   VAE                      0.892                 0.000                0.6           0.7                562.2                0.020          0.548
   GAN                      0.753                 0.223                0.3           0.7                621.5                0.001          0.494
  cGAN                      0.781                 0.162                0.9           0.7                626.0                0.078          0.636
  DDPM                      0.757                 0.987                0.8           0.7               1371.4                3.269          0.811
```

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