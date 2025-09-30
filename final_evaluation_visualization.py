import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def create_sample_data():
    """Create sample performance data for four models across four metrics."""
    models = ["VAE", "GAN", "cGAN", "DDPM"]

    # Normalized metrics (0-1 scale, where 1 is best)
    data = {
        "VAE": {
            "Image Quality": 0.7,
            "Training Stability": 0.9,
            "Controllability": 0.6,
            "Efficiency": 0.8,
        },
        "GAN": {
            "Image Quality": 0.8,
            "Training Stability": 0.5,
            "Controllability": 0.7,
            "Efficiency": 0.6,
        },
        "cGAN": {
            "Image Quality": 0.85,
            "Training Stability": 0.6,
            "Controllability": 0.9,
            "Efficiency": 0.7,
        },
        "DDPM": {
            "Image Quality": 0.95,
            "Training Stability": 0.8,
            "Controllability": 0.8,
            "Efficiency": 0.4,
        },
    }

    return models, data


def plot_radar_chart(models, data):
    """Create an optimized radar chart with clear color scheme."""
    metrics = list(data[models[0]].keys())
    num_metrics = len(metrics)

    # Compute angle for each metric
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]  # Complete the circle

    # Set up the plot with high DPI for clarity
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection="polar"), dpi=100
    )

    # Define colors with high contrast and accessibility
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
    line_styles = ["-", "--", "-.", ":"]

    # Plot each model
    for i, model in enumerate(models):
        values = [data[model][metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2.5,
            label=model,
            color=colors[i],
            linestyle=line_styles[i],
            markersize=8,
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add legend with better positioning
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title(
        "Generative Models Performance Comparison\n(Higher values = Better performance)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    return fig


def plot_3d_spherical_zones(models, data):
    """Create 3D scatter plot with spherical performance zones."""
    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Extract 3D coordinates (using first 3 metrics)
    metrics_3d = ["Image Quality", "Training Stability", "Controllability"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot model points
    for i, model in enumerate(models):
        x = data[model][metrics_3d[0]]
        y = data[model][metrics_3d[1]]
        z = data[model][metrics_3d[2]]

        ax.scatter(
            x,
            y,
            z,
            s=200,
            c=colors[i],
            label=model,
            edgecolors="black",
            linewidth=2,
            alpha=0.8,
        )

        # Add model labels near points
        ax.text(x, y, z + 0.05, model, fontsize=12, fontweight="bold")

    # Create spherical performance zones
    # Best performance is at corner (1,1,1)
    # Distance from ideal = sqrt((x-1)² + (y-1)² + (z-1)²)
    # Max distance = sqrt(3) ≈ 1.732

    # Create a grid for visualization
    grid_points = np.linspace(0, 1, 20)
    X, Y, Z = np.meshgrid(grid_points, grid_points, grid_points)

    # Calculate distance from ideal point (1,1,1)
    distances = np.sqrt((X - 1) ** 2 + (Y - 1) ** 2 + (Z - 1) ** 2)
    max_distance = np.sqrt(3)

    # Performance score: 1 - (distance / max_distance)
    performance_scores = 1 - (distances / max_distance)

    # Create spherical performance zones as actual semi-transparent spheres
    # Each sphere represents a performance level at distance from ideal point (1,1,1)
    performance_levels = [0.9, 0.7, 0.5, 0.3]
    zone_colors = ["#00AA00", "#66BB00", "#FFAA00", "#FF6666"]  # Green to red
    zone_alphas = [0.15, 0.20, 0.25, 0.30]  # Semi-transparent

    # Calculate radius for each performance level
    # Performance = 1 - (distance / max_distance)
    # So distance = (1 - performance) * max_distance
    max_distance = np.sqrt(3)

    for level, color, alpha in zip(performance_levels, zone_colors, zone_alphas):
        # Calculate radius for this performance level
        radius = (1 - level) * max_distance

        # Create sphere surface
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        phi, theta = np.meshgrid(phi, theta)

        # Sphere coordinates centered at (1,1,1)
        x_sphere = 1 + radius * np.sin(phi) * np.cos(theta)
        y_sphere = 1 + radius * np.sin(phi) * np.sin(theta)
        z_sphere = 1 + radius * np.cos(phi)

        # Only plot the part of sphere within the [0,1] cube
        mask = (
            (x_sphere >= 0)
            & (x_sphere <= 1)
            & (y_sphere >= 0)
            & (y_sphere <= 1)
            & (z_sphere >= 0)
            & (z_sphere <= 1)
        )

        # Apply mask to sphere coordinates
        x_masked = np.where(mask, x_sphere, np.nan)
        y_masked = np.where(mask, y_sphere, np.nan)
        z_masked = np.where(mask, z_sphere, np.nan)

        ax.plot_surface(
            x_masked,
            y_masked,
            z_masked,
            color=color,
            alpha=alpha,
            label=f"Performance ≥ {level:.1f}",
            linewidth=0,
            antialiased=True,
        )

        # Add colored volume inside sphere with very low transparency
        # Create scattered points inside the sphere for volume effect
        n_volume_points = 800

        # Generate random points in a cube around (1,1,1)
        x_vol = np.random.uniform(
            max(0, 1 - radius), min(1, 1 + radius), n_volume_points
        )
        y_vol = np.random.uniform(
            max(0, 1 - radius), min(1, 1 + radius), n_volume_points
        )
        z_vol = np.random.uniform(
            max(0, 1 - radius), min(1, 1 + radius), n_volume_points
        )

        # Keep only points inside the sphere and within [0,1] cube
        distances_vol = np.sqrt((x_vol - 1) ** 2 + (y_vol - 1) ** 2 + (z_vol - 1) ** 2)
        inside_sphere = distances_vol <= radius
        inside_cube = (
            (x_vol >= 0)
            & (x_vol <= 1)
            & (y_vol >= 0)
            & (y_vol <= 1)
            & (z_vol >= 0)
            & (z_vol <= 1)
        )

        valid_points = inside_sphere & inside_cube

        if np.any(valid_points):
            ax.scatter(
                x_vol[valid_points],
                y_vol[valid_points],
                z_vol[valid_points],
                c=color,
                s=0.5,
                alpha=0.12,  # Very low transparency for volume
                edgecolors="none",
            )

    # Highlight the ideal corner
    ax.scatter(
        [1],
        [1],
        [1],
        s=300,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidth=2,
        label="Ideal Performance",
    )

    # Customize the plot
    ax.set_xlabel("Image Quality →", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Stability →", fontsize=12, fontweight="bold")
    ax.set_zlabel("Controllability →", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Set ticks
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Legend
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10)

    plt.title(
        "3D Performance Space with Spherical Zones\n(Closer to upper-right corner = Better overall performance)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_performance_heatmap(models, data):
    """Create a performance heatmap with optimized colors."""
    metrics = list(data[models[0]].keys())

    # Create matrix
    matrix = np.array([[data[model][metric] for metric in metrics] for model in models])

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Use a perceptually uniform colormap
    sns.heatmap(
        matrix,
        xticklabels=metrics,
        yticklabels=models,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Red-Yellow-Green (intuitive: red=bad, green=good)
        center=0.5,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Performance Score (0-1)"},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    plt.title(
        "Performance Heatmap: Models vs Metrics\n(Green = Better, Red = Worse)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    plt.ylabel("Generative Models", fontsize=12, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


def create_comprehensive_visualization():
    """Create all visualizations with optimized color schemes."""
    models, data = create_sample_data()

    print("Creating optimized visualizations...")
    print("=" * 50)

    # 1. Radar Chart (Best for overall comparison)
    print("1. Creating Radar Chart...")
    fig1 = plot_radar_chart(models, data)
    fig1.savefig("radar_chart_optimized.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: radar_chart_optimized.png")

    # 2. 3D Spherical Zones
    print("2. Creating 3D Spherical Performance Zones...")
    fig2 = plot_3d_spherical_zones(models, data)
    fig2.savefig("3d_spherical_zones.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: 3d_spherical_zones.png")

    # 3. Performance Heatmap
    print("3. Creating Performance Heatmap...")
    fig3 = plot_performance_heatmap(models, data)
    fig3.savefig("performance_heatmap.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: performance_heatmap.png")

    plt.show()

    print("=" * 50)
    print("All visualizations created successfully!")
    print("\nVisualization Guide:")
    print("• Radar Chart: Best for comparing overall model performance")
    print("• 3D Spherical Zones: Shows performance regions in 3D space")
    print("• Heatmap: Clear numerical comparison across all metrics")
    print("\nColor Schemes:")
    print("• Radar Chart: Distinct colors with transparency for overlap visibility")
    print("• 3D Plot: Spherical zones from ideal corner (1,1,1)")
    print("• Heatmap: Red-Yellow-Green gradient (intuitive good/bad mapping)")


def print_performance_summary(models, data):
    """Print a summary of model performances."""
    print("\nPerformance Summary:")
    print("=" * 60)

    for model in models:
        metrics = data[model]
        avg_score = sum(metrics.values()) / len(metrics)
        print(f"\n{model}:")
        print(f"  Average Score: {avg_score:.3f}")

        # Calculate distance from ideal (for 3D visualization)
        coords = [
            metrics["Image Quality"],
            metrics["Training Stability"],
            metrics["Controllability"],
        ]
        distance = np.sqrt(sum((x - 1) ** 2 for x in coords))
        spherical_score = 1 - (distance / np.sqrt(3))
        print(f"  3D Spherical Score: {spherical_score:.3f}")

        # Individual metrics
        for metric, score in metrics.items():
            print(f"    {metric}: {score:.2f}")


if __name__ == "__main__":
    models, data = create_sample_data()
    print_performance_summary(models, data)
    create_comprehensive_visualization()
