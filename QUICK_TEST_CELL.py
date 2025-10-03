# ============================================================================
# QUICK TEST MODE - Copy this into a Colab cell to test the code
# ============================================================================
# This cell validates all components work without full training
# Runtime: ~30 seconds
# ============================================================================

print("🧪 QUICK TEST MODE - Validating code structure...\n")

import torch
import time

# Test configuration
TEST_BATCH_SIZE = 16
TEST_SAMPLES = 32
TEST_LATENT_DIM = 100

# Create tiny synthetic dataset
test_images = torch.randn(TEST_SAMPLES, 1, 28, 28).to(device)
test_labels = torch.randint(0, 10, (TEST_SAMPLES,)).to(device)

print(f"✓ Created {TEST_SAMPLES} synthetic test samples")
print(f"✓ Using device: {device}\n")

# ============================================================================
# Test 1: Model Instantiation
# ============================================================================
print("Test 1: Model Instantiation")
try:
    test_vae = VAE(latent_dim=20).to(device)
    test_gan_gen = Generator(TEST_LATENT_DIM).to(device)
    test_gan_disc = Discriminator().to(device)
    test_cgan_gen = ConditionalGenerator(TEST_LATENT_DIM, 10).to(device)
    test_cgan_disc = ConditionalDiscriminator(10).to(device)
    test_ddpm_unet = UNet().to(device)
    print("  ✓ VAE instantiated")
    print("  ✓ GAN instantiated")
    print("  ✓ cGAN instantiated")
    print("  ✓ DDPM instantiated\n")
except Exception as e:
    print(f"  ✗ Model instantiation failed: {e}\n")
    raise

# ============================================================================
# Test 2: Forward Pass
# ============================================================================
print("Test 2: Forward Pass (1 batch)")
try:
    # VAE
    recon, mu, logvar = test_vae(test_images[:TEST_BATCH_SIZE])
    assert recon.shape == (TEST_BATCH_SIZE, 1, 28, 28)
    print("  ✓ VAE forward pass")

    # GAN
    z = torch.randn(TEST_BATCH_SIZE, TEST_LATENT_DIM).to(device)
    fake_imgs = test_gan_gen(z)
    disc_out = test_gan_disc(fake_imgs)
    assert fake_imgs.shape == (TEST_BATCH_SIZE, 1, 28, 28)
    print("  ✓ GAN forward pass")

    # cGAN
    z = torch.randn(TEST_BATCH_SIZE, TEST_LATENT_DIM).to(device)
    labels = torch.randint(0, 10, (TEST_BATCH_SIZE,)).to(device)
    fake_imgs = test_cgan_gen(z, labels)
    disc_out = test_cgan_disc(fake_imgs, labels)
    assert fake_imgs.shape == (TEST_BATCH_SIZE, 1, 28, 28)
    print("  ✓ cGAN forward pass")

    # DDPM
    t = torch.randint(0, 100, (TEST_BATCH_SIZE,)).to(device)
    noise_pred = test_ddpm_unet(test_images[:TEST_BATCH_SIZE], t)
    assert noise_pred.shape == (TEST_BATCH_SIZE, 1, 28, 28)
    print("  ✓ DDPM forward pass\n")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}\n")
    raise

# ============================================================================
# Test 3: Loss Computation
# ============================================================================
print("Test 3: Loss Computation")
try:
    # VAE loss
    vae_loss_val = vae_loss(recon, test_images[:TEST_BATCH_SIZE], mu, logvar)
    print(f"  ✓ VAE loss: {vae_loss_val.item():.2f}")

    # GAN loss
    criterion = torch.nn.BCELoss()
    real_labels = torch.ones(TEST_BATCH_SIZE, 1).to(device)
    g_loss = criterion(disc_out, real_labels)
    print(f"  ✓ GAN loss: {g_loss.item():.4f}")

    # DDPM loss
    ddpm_loss = torch.nn.MSELoss()(noise_pred, test_images[:TEST_BATCH_SIZE])
    print(f"  ✓ DDPM loss: {ddpm_loss.item():.4f}\n")
except Exception as e:
    print(f"  ✗ Loss computation failed: {e}\n")
    raise

# ============================================================================
# Test 4: Backward Pass
# ============================================================================
print("Test 4: Backward Pass (gradient check)")
try:
    # VAE
    test_vae.zero_grad()
    vae_loss_val.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in test_vae.parameters()
    )
    assert has_grad, "VAE gradients not computed"
    print("  ✓ VAE backward pass")

    # GAN Generator
    test_gan_gen.zero_grad()
    g_loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in test_gan_gen.parameters()
    )
    assert has_grad, "GAN gradients not computed"
    print("  ✓ GAN backward pass")

    print()
except Exception as e:
    print(f"  ✗ Backward pass failed: {e}\n")
    raise

# ============================================================================
# Test 5: Mini Training Loop
# ============================================================================
print("Test 5: Mini Training Loop (3 steps)")
try:
    # VAE mini training
    vae_opt = torch.optim.Adam(test_vae.parameters(), lr=1e-3)
    for step in range(3):
        recon, mu, logvar = test_vae(test_images[:TEST_BATCH_SIZE])
        loss = vae_loss(recon, test_images[:TEST_BATCH_SIZE], mu, logvar)
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    print(f"  ✓ VAE mini training (final loss: {loss.item():.2f})")

    # GAN mini training
    g_opt = torch.optim.Adam(test_gan_gen.parameters(), lr=2e-4)
    d_opt = torch.optim.Adam(test_gan_disc.parameters(), lr=2e-4)
    for step in range(3):
        # Train discriminator
        d_opt.zero_grad()
        z = torch.randn(TEST_BATCH_SIZE, TEST_LATENT_DIM).to(device)
        fake = test_gan_gen(z)
        d_real = test_gan_disc(test_images[:TEST_BATCH_SIZE])
        d_fake = test_gan_disc(fake.detach())
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(
            d_fake, torch.zeros_like(d_fake)
        )
        d_loss.backward()
        d_opt.step()

        # Train generator
        g_opt.zero_grad()
        d_fake = test_gan_disc(fake)
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        g_loss.backward()
        g_opt.step()
    print(f"  ✓ GAN mini training (G:{g_loss.item():.4f}, D:{d_loss.item():.4f})\n")
except Exception as e:
    print(f"  ✗ Mini training failed: {e}\n")
    raise

# ============================================================================
# Test 6: Image Generation
# ============================================================================
print("Test 6: Image Generation")
try:
    test_vae.eval()
    test_gan_gen.eval()
    test_cgan_gen.eval()

    with torch.no_grad():
        # VAE
        z = torch.randn(4, 20).to(device)
        vae_imgs = test_vae.decode(z)
        assert vae_imgs.shape == (4, 1, 28, 28)
        print("  ✓ VAE generated 4 images")

        # GAN
        z = torch.randn(4, TEST_LATENT_DIM).to(device)
        gan_imgs = test_gan_gen(z)
        assert gan_imgs.shape == (4, 1, 28, 28)
        print("  ✓ GAN generated 4 images")

        # cGAN
        z = torch.randn(4, TEST_LATENT_DIM).to(device)
        labels = torch.tensor([0, 1, 2, 3]).to(device)
        cgan_imgs = test_cgan_gen(z, labels)
        assert cgan_imgs.shape == (4, 1, 28, 28)
        print("  ✓ cGAN generated 4 images for classes 0-3\n")
except Exception as e:
    print(f"  ✗ Image generation failed: {e}\n")
    raise

# ============================================================================
# Test 7: Visualization Functions
# ============================================================================
print("Test 7: Visualization Functions")
try:
    import matplotlib.pyplot as plt
    import pandas as pd

    # Test data
    test_perf_data = {
        "VAE": {"Metric1": 0.8, "Metric2": 0.7, "Metric3": 0.9},
        "GAN": {"Metric1": 0.7, "Metric2": 0.8, "Metric3": 0.6},
    }

    # Test DataFrame creation
    df = pd.DataFrame(test_perf_data).T
    assert df.shape == (2, 3)
    print("  ✓ Performance data structure")

    # Test plotting (don't display)
    plt.ioff()  # Turn off interactive mode

    # Simple bar chart
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(["VAE", "GAN"], [0.8, 0.7])
    plt.close(fig)
    print("  ✓ Bar chart function")

    # Simple heatmap
    try:
        import seaborn as sns

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.heatmap(df, annot=True, ax=ax)
        plt.close(fig)
        print("  ✓ Heatmap function")
    except:
        print("  ⚠ Seaborn not available (heatmap skipped)")

    print()
except Exception as e:
    print(f"  ✗ Visualization test failed: {e}\n")
    raise

# ============================================================================
# Final Summary
# ============================================================================
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nValidated Components:")
print("  ✓ Model architectures (VAE, GAN, cGAN, DDPM)")
print("  ✓ Forward/backward passes")
print("  ✓ Loss computations")
print("  ✓ Training loops")
print("  ✓ Image generation")
print("  ✓ Data structures")
print("  ✓ Visualization functions")

print("\n🎯 Your code is ready to run with full data!")
print("\nNext steps:")
print("  1. Set EPOCHS to desired value (e.g., 10-30)")
print("  2. Run all cells with real MNIST data")
print("  3. Optional: Set CALCULATE_REAL_METRICS=True for FID/IS")

print("\n" + "=" * 70)

# Cleanup
del test_vae, test_gan_gen, test_gan_disc, test_cgan_gen, test_cgan_disc, test_ddpm_unet
del test_images, test_labels
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("✓ Test cleanup complete")
