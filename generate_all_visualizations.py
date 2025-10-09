"""
Generate all visualizations with actual evaluation metrics.
This script creates comprehensive visualizations for MNIST generative models comparison.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import plotly.graph_objects as go
import os

# Create output directory
os.makedirs("outputs/visualizations", exist_ok=True)

# Actual evaluation metrics from user
metrics = {
    "VAE": {
        "FID": 37.56,
        "IS_mean": 2.39,
        "IS_std": 0.09,
        "time": 0.19,
        "quality": 0.483411,
        "stability": 0.8,
        "controllability": 0.170799,
        "efficiency": 0.999095,
    },
    "GAN": {
        "FID": 77.91,
        "IS_mean": 3.04,
        "IS_std": 0.17,
        "time": 0.02,
        "quality": 0.418509,
        "stability": 0.6,
        "controllability": 0.065772,
        "efficiency": 0.999915,
    },
    "cGAN": {
        "FID": 71.09,
        "IS_mean": 2.81,
        "IS_std": 0.20,
        "time": 0.04,
        "quality": 0.422947,
        "stability": 0.6,
        "controllability": 0.962000,
        "efficiency": 0.999807,
    },
    "DDPM": {
        "FID": 53.15,
        "IS_mean": 2.17,
        "IS_std": 0.08,
        "time": 215.05,
        "quality": 0.432172,
        "stability": 0.7,
        "controllability": 0.691701,
        "efficiency": 0.000000,
    },
}

model_names = list(metrics.keys())
colors = {"VAE": "#1f77b4", "GAN": "#ff7f0e", "cGAN": "#2ca02c", "DDPM": "#d62728"}

print("Generating visualizations with actual evaluation metrics...")

# ============================================================================
# 1. RADAR CHART - Performance Scores
# ============================================================================
print("1. Creating radar chart...")

categories = ["Image Quality", "Training\nStability", "Controllability", "Efficiency"]
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="polar"))

for model in model_names:
    values = [
        metrics[model]["quality"],
        metrics[model]["stability"],
        metrics[model]["controllability"],
        metrics[model]["efficiency"],
    ]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[model])
    ax.fill(angles, values, alpha=0.15, color=colors[model])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9)
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.title("Performance Comparison Across All Metrics", size=14, weight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/visualizations/radar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================================
# 2. HEATMAP - Normalized Performance Scores
# ============================================================================
print("2. Creating performance heatmap...")

performance_data = np.array(
    [
        [metrics[model]["quality"] for model in model_names],
        [metrics[model]["stability"] for model in model_names],
        [metrics[model]["controllability"] for model in model_names],
        [metrics[model]["efficiency"] for model in model_names],
    ]
)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(performance_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(model_names)))
ax.set_yticks(np.arange(len(categories)))
ax.set_xticklabels(model_names, fontsize=11)
ax.set_yticklabels(categories, fontsize=11)

for i in range(len(categories)):
    for j in range(len(model_names)):
        text = ax.text(
            j,
            i,
            f"{performance_data[i, j]:.3f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            weight="bold",
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Normalized Score", rotation=270, labelpad=20, fontsize=11)
plt.title("Normalized Performance Heatmap", fontsize=14, weight="bold", pad=15)
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/performance_heatmap.png", dpi=300, bbox_inches="tight"
)
plt.close()

# ============================================================================
# 3. BAR CHART - FID Scores (Lower is Better)
# ============================================================================
print("3. Creating FID comparison bar chart...")

fid_scores = [metrics[model]["FID"] for model in model_names]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    model_names,
    fid_scores,
    color=[colors[m] for m in model_names],
    alpha=0.8,
    edgecolor="black",
)

for bar, score in zip(bars, fid_scores):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{score:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

ax.set_ylabel("FID Score (Lower is Better)", fontsize=12, weight="bold")
ax.set_title(
    "Frechet Inception Distance (FID) Comparison", fontsize=14, weight="bold", pad=15
)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("outputs/visualizations/fid_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================================
# 4. BAR CHART - Inception Score (Higher is Better)
# ============================================================================
print("4. Creating Inception Score comparison bar chart...")

is_means = [metrics[model]["IS_mean"] for model in model_names]
is_stds = [metrics[model]["IS_std"] for model in model_names]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    model_names,
    is_means,
    yerr=is_stds,
    color=[colors[m] for m in model_names],
    alpha=0.8,
    edgecolor="black",
    capsize=5,
    error_kw={"linewidth": 2},
)

for bar, mean, std in zip(bars, is_means, is_stds):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + std,
        f"{mean:.2f}±{std:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

ax.set_ylabel("Inception Score (Higher is Better)", fontsize=12, weight="bold")
ax.set_title("Inception Score (IS) Comparison", fontsize=14, weight="bold", pad=15)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/inception_score_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================================
# 5. BAR CHART - Generation Time
# ============================================================================
print("5. Creating generation time comparison bar chart...")

times = [metrics[model]["time"] for model in model_names]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    model_names,
    times,
    color=[colors[m] for m in model_names],
    alpha=0.8,
    edgecolor="black",
)

for bar, time in zip(bars, times):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{time:.2f}s",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

ax.set_ylabel("Generation Time (seconds)", fontsize=12, weight="bold")
ax.set_title("Generation Time for 1000 Samples", fontsize=14, weight="bold", pad=15)
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/generation_time_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================================
# 6. 3D PERFORMANCE VISUALIZATION - Static with Coordinates
# ============================================================================
print("6. Creating 3D performance visualization (static)...")


def plot_full_cuboid(ax, x_min, y_min, z_min, color, alpha, label):
    """Plot a filled cuboid zone extending from (x_min, y_min, z_min) to (1.0, 1.0, 1.0)."""
    x_range, y_range, z_range = [x_min, 1.0], [y_min, 1.0], [z_min, 1.0]

    # Plot all 6 faces of the cuboid
    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_surface(xx, yy, np.full_like(xx, z_min), color=color, alpha=alpha)
    ax.plot_surface(xx, yy, np.full_like(xx, 1.0), color=color, alpha=alpha)

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_surface(xx, np.full_like(xx, y_min), zz, color=color, alpha=alpha)
    ax.plot_surface(xx, np.full_like(xx, 1.0), zz, color=color, alpha=alpha)

    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_surface(np.full_like(yy, x_min), yy, zz, color=color, alpha=alpha)
    ax.plot_surface(np.full_like(yy, 1.0), yy, zz, color=color, alpha=alpha)

    return mpatches.Patch(facecolor=color, alpha=0.6, label=label)


fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Performance zones (extending from min thresholds to 1.0, matching notebook)
zones = {
    "Elite": ((0.9, 0.85, 0.8), "#2ECC71", 0.1),
    "Excellent": ((0.8, 0.7, 0.6), "#3498DB", 0.1),
    "Good": ((0.6, 0.5, 0.4), "#F39C12", 0.05),
}

legend_patches = []
for label, ((x, y, z), color, alpha) in sorted(
    zones.items(), key=lambda item: item[1][0][0]
):
    patch = plot_full_cuboid(
        ax, x, y, z, color, alpha, f"{label} (Q≥{x}, S≥{y}, C≥{z})"
    )
    legend_patches.append(patch)

# Plot model points (using notebook colors)
model_colors_3d = {
    "VAE": "#5D6D7E",
    "GAN": "#E74C3C",
    "cGAN": "#2ECC71",
    "DDPM": "#F39C12",
}
for model in model_names:
    x, y, z = (
        metrics[model]["quality"],
        metrics[model]["stability"],
        metrics[model]["controllability"],
    )
    ax.scatter(
        x,
        y,
        z,
        c=model_colors_3d[model],
        s=400,
        marker="o",
        edgecolors="black",
        linewidths=2.5,
        label=model,
        zorder=20,
    )

    # Add model name and coordinates label
    ax.text(
        x,
        y,
        z + 0.05,
        f"  {model}\n  ({x:.2f}, {y:.2f}, {z:.2f})",
        fontsize=12,
        weight="bold",
        zorder=21,
        ha="left",
    )

# Add ideal point (1.0, 1.0, 1.0)
ax.scatter(
    1,
    1,
    1,
    c="#34495E",
    s=600,
    marker="*",
    edgecolors="gold",
    linewidths=2.5,
    label="Ideal (1.0)",
    zorder=25,
)

ax.set_xlabel("\nImage Quality", fontsize=16, labelpad=25)
ax.set_ylabel("\nTraining Stability", fontsize=16, labelpad=25)
ax.set_zlabel("\nControllability", fontsize=16, labelpad=25)
ax.set_title(
    "3D Performance Space with Filled Cuboid Zones",
    fontsize=24,
    weight="bold",
    pad=30,
)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_zlim(0, 1.0)
ax.view_init(elev=28, azim=-50)

# Build legend with zones first, then models
handles, _ = ax.get_legend_handles_labels()
ax.legend(
    handles=list(reversed(legend_patches)) + handles,
    loc="upper left",
    bbox_to_anchor=(-0.1, 1.0),
    fontsize=12,
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    edgecolor="black",
    borderpad=1,
    title_fontsize=14,
    title="Legend",
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "outputs/visualizations/3d_performance_static.png", dpi=300, bbox_inches="tight"
)
plt.close()

# ============================================================================
# 7. 3D PERFORMANCE VISUALIZATION - Interactive (Plotly)
# ============================================================================
print("7. Creating 3D performance visualization (interactive)...")


def create_cuboid_mesh(x_min, y_min, z_min, color, name, opacity=0.15):
    """Create a cuboid mesh for Plotly extending from (x_min, y_min, z_min) to (1.0, 1.0, 1.0)."""
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [1.0, y_min, z_min],
            [1.0, 1.0, z_min],
            [x_min, 1.0, z_min],
            [x_min, y_min, 1.0],
            [1.0, y_min, 1.0],
            [1.0, 1.0, 1.0],
            [x_min, 1.0, 1.0],
        ]
    )

    faces = np.array(
        [
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    i, j, k = [], [], []
    for face in faces:
        i.extend([face[0], face[0]])
        j.extend([face[1], face[2]])
        k.extend([face[2], face[3]])

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


# Create figure
fig = go.Figure()

# Add performance zones (matching notebook: from min thresholds to 1.0)
fig.add_trace(
    create_cuboid_mesh(0.9, 0.85, 0.8, "#2ECC71", "Elite (Q≥0.9, S≥0.85, C≥0.8)", 0.1)
)
fig.add_trace(
    create_cuboid_mesh(0.8, 0.7, 0.6, "#3498DB", "Excellent (Q≥0.8, S≥0.7, C≥0.6)", 0.1)
)
fig.add_trace(
    create_cuboid_mesh(0.6, 0.5, 0.4, "#F39C12", "Good (Q≥0.6, S≥0.5, C≥0.4)", 0.05)
)

# Add model points (using notebook colors)
for model in model_names:
    x, y, z = (
        metrics[model]["quality"],
        metrics[model]["stability"],
        metrics[model]["controllability"],
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers+text",
            marker=dict(
                size=12, color=model_colors_3d[model], line=dict(color="black", width=2)
            ),
            text=[model],
            textposition="top center",
            textfont=dict(size=12, color="black", family="Arial Black"),
            name=model,
            hovertemplate=(
                f"<b>{model}</b><br>"
                f"Quality: {x:.3f}<br>"
                f"Stability: {y:.3f}<br>"
                f"Controllability: {z:.3f}<br>"
                f"FID: {metrics[model]['FID']:.2f}<br>"
                f"IS: {metrics[model]['IS_mean']:.2f}±{metrics[model]['IS_std']:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

# Add ideal point
fig.add_trace(
    go.Scatter3d(
        x=[1.0],
        y=[1.0],
        z=[1.0],
        mode="markers",
        marker=dict(
            size=15, color="#34495E", symbol="diamond", line=dict(color="gold", width=3)
        ),
        name="Ideal (1.0, 1.0, 1.0)",
        hovertemplate="<b>Ideal Point</b><br>Perfect Score (1.0, 1.0, 1.0)<extra></extra>",
    )
)

# Update layout
fig.update_layout(
    title=dict(
        text="3D Performance Landscape - Interactive<br><sub>Quality × Stability × Controllability</sub>",
        font=dict(size=16, color="black", family="Arial Black"),
        x=0.5,
        xanchor="center",
    ),
    scene=dict(
        xaxis=dict(title="Image Quality", range=[0, 1], showgrid=True),
        yaxis=dict(title="Training Stability", range=[0, 1], showgrid=True),
        zaxis=dict(title="Controllability", range=[0, 1], showgrid=True),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    ),
    showlegend=True,
    legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    width=1000,
    height=800,
)

fig.write_html("outputs/visualizations/3d_performance_interactive.html")

# ============================================================================
# 8. SUMMARY METRICS TABLE VISUALIZATION
# ============================================================================
print("8. Creating summary metrics table...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("tight")
ax.axis("off")

table_data = []
for model in model_names:
    table_data.append(
        [
            model,
            f"{metrics[model]['FID']:.2f}",
            f"{metrics[model]['IS_mean']:.2f} ± {metrics[model]['IS_std']:.2f}",
            f"{metrics[model]['quality']:.3f}",
            f"{metrics[model]['stability']:.3f}",
            f"{metrics[model]['controllability']:.3f}",
            f"{metrics[model]['efficiency']:.3f}",
            f"{metrics[model]['time']:.2f}s",
        ]
    )

table = ax.table(
    cellText=table_data,
    colLabels=[
        "Model",
        "FID↓",
        "Inception Score↑",
        "Quality",
        "Stability",
        "Control.",
        "Efficiency",
        "Time",
    ],
    cellLoc="center",
    loc="center",
    colWidths=[0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1],
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color header
for i in range(8):
    table[(0, i)].set_facecolor("#4472C4")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Color rows by model
for i, model in enumerate(model_names, start=1):
    table[(i, 0)].set_facecolor(colors[model])
    table[(i, 0)].set_text_props(weight="bold", color="white")

plt.title("Comprehensive Metrics Summary", fontsize=14, weight="bold", pad=20)
plt.savefig(
    "outputs/visualizations/metrics_summary_table.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files in outputs/visualizations/:")
print("  1. radar_chart.png - Performance comparison across all metrics")
print("  2. performance_heatmap.png - Normalized performance scores")
print("  3. fid_comparison.png - FID scores comparison")
print("  4. inception_score_comparison.png - IS scores with error bars")
print("  5. generation_time_comparison.png - Generation time (log scale)")
print("  6. 3d_performance_static.png - 3D visualization with coordinates")
print("  7. 3d_performance_interactive.html - Interactive 3D (open in browser)")
print("  8. metrics_summary_table.png - Comprehensive metrics table")
print("=" * 70)
