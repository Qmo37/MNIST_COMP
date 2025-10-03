#!/usr/bin/env python3
"""
Update 3D visualizations to follow research best practices.
Removes arbitrary quality zones and uses quantitative metrics only.
"""

import json
import sys

notebook_path = "MNIST_Generative_Models_Complete.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Updated Plotly function
plotly_function = '''def create_interactive_3d_spherical_zone_colab(
    performance_data,
    save_path_html="outputs/visualizations/3d_spherical_interactive.html",
    save_path_png="outputs/visualizations/3d_spherical_zone.png",
):
    """
    Create interactive 3D performance visualization following research best practices.

    Features:
    - Model positions shown in normalized [0,1] metric space
    - Distance-to-ideal visualization showing proximity to perfect performance
    - No arbitrary quality zones - uses quantitative comparison only
    - Interactive in Colab/Jupyter, also saves HTML and PNG

    Args:
        performance_data: Dict with model names as keys and metric dicts as values
        save_path_html: Path to save interactive HTML
        save_path_png: Path to save static PNG
    """

    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available. Creating static visualization only.")
        create_static_3d_spherical_zone(performance_data, save_path_png)
        return None

    # Colors for each model
    colors_dict = {
        "VAE": "#6B9BD1",  # Blue
        "GAN": "#F4A460",  # Orange
        "cGAN": "#90EE90",  # Light green
        "DDPM": "#CD5C5C",  # Red
    }

    # Create plotly figure
    fig = go.Figure()

    # Add a subtle gradient sphere showing distance to ideal point (1,1,1)
    # This provides visual reference without arbitrary thresholds
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 15)

    # Sphere centered at ideal point (1,1,1)
    radius = 0.8
    x_sphere = 1.0 - radius * 0.5 + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = 1.0 - radius * 0.5 + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = 1.0 - radius * 0.5 + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Clip to valid range
    x_sphere = np.clip(x_sphere, 0, 1)
    y_sphere = np.clip(y_sphere, 0, 1)
    z_sphere = np.clip(z_sphere, 0, 1)

    # Calculate distance from ideal for gradient coloring
    distances = np.sqrt((x_sphere - 1)**2 + (y_sphere - 1)**2 + (z_sphere - 1)**2)

    # Add subtle reference surface
    fig.add_trace(
        go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            surfacecolor=distances,
            colorscale=[
                [0, "rgba(150, 255, 150, 0.08)"],  # Near ideal
                [0.7, "rgba(255, 255, 150, 0.06)"],  # Medium distance
                [1, "rgba(255, 150, 150, 0.04)"]  # Far from ideal
            ],
            showscale=False,
            opacity=0.2,
            name="Reference Gradient",
            hovertemplate="<b>Distance to Ideal (1,1,1)</b><br>" +
                          "Closer = Better Overall Performance<br>" +
                          "<extra></extra>",
            showlegend=True
        )
    )

    # Process each model and add as scatter points
    for model_name, metrics in performance_data.items():
        metrics_list = list(metrics.values())

        if len(metrics_list) >= 3:
            # Get first 3 metrics for 3D coordinates
            x = metrics_list[0]  # Image Quality
            y = metrics_list[1]  # Training Stability
            z = metrics_list[2]  # Controllability

            # Calculate distance to ideal for reference
            distance_to_ideal = np.sqrt((x - 1)**2 + (y - 1)**2 + (z - 1)**2)
            avg_score = (x + y + z) / 3

            # Add model as scatter point
            fig.add_trace(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers+text",
                    marker=dict(
                        size=16,
                        color=colors_dict.get(model_name, "#333333"),
                        symbol="circle",
                        line=dict(color="black", width=2.5),
                    ),
                    text=[model_name],
                    textposition="top center",
                    textfont=dict(size=14, color="black", family="Arial", weight="bold"),
                    name=model_name,
                    hovertemplate=f"<b>{model_name}</b><br>" +
                    f"Image Quality: {x:.3f}<br>" +
                    f"Training Stability: {y:.3f}<br>" +
                    f"Controllability: {z:.3f}<br>" +
                    f"Average Score: {avg_score:.3f}<br>" +
                    f"Distance to Ideal: {distance_to_ideal:.3f}<br>" +
                    "<extra></extra>",
                )
            )

    # Add ideal performance indicator at (1,1,1)
    fig.add_trace(
        go.Scatter3d(
            x=[1.0],
            y=[1.0],
            z=[1.0],
            mode="markers+text",
            marker=dict(
                size=22,
                color="gold",
                symbol="diamond",
                line=dict(color="black", width=3)
            ),
            text=["★ Ideal"],
            textposition="top center",
            textfont=dict(size=16, color="black", family="Arial", weight="bold"),
            name="Ideal Performance",
            hovertemplate="<b>Ideal Performance Point</b><br>" +
            "All metrics = 1.0<br>" +
            "Target for optimization<br>" +
            "<extra></extra>",
        )
    )

    # Update layout with research-appropriate title
    fig.update_layout(
        title={
            "text": "3D Performance Space: Normalized Metrics Comparison<br>" +
            "<sub>(All metrics normalized to [0,1], higher = better)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 20, "family": "Arial"},
        },
        scene=dict(
            xaxis=dict(
                title="Image Quality (Normalized) →",
                titlefont=dict(size=14, family="Arial"),
                range=[0, 1.05],
                showgrid=True,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(240, 240, 245, 0.3)",
            ),
            yaxis=dict(
                title="Training Stability (Normalized) →",
                titlefont=dict(size=14, family="Arial"),
                range=[0, 1.05],
                showgrid=True,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(240, 245, 240, 0.3)",
            ),
            zaxis=dict(
                title="Controllability (Normalized) ↑",
                titlefont=dict(size=14, family="Arial"),
                range=[0, 1.05],
                showgrid=True,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(245, 240, 240, 0.3)",
            ),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.4)),
            aspectmode="cube",
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11, family="Arial"),
        ),
        width=1000,
        height=900,
        margin=dict(l=10, r=10, t=120, b=10),
        hovermode="closest",
    )

    # Save interactive HTML
    try:
        fig.write_html(save_path_html)
        print(f"✅ Interactive HTML saved to: {save_path_html}")
        print(f"   Open this file in a browser for full interactivity!")
    except Exception as e:
        print(f"⚠️ Could not save HTML: {e}")

    # Also create static matplotlib version
    create_static_3d_spherical_zone(performance_data, save_path_png)

    # Display in Colab/Jupyter
    print("\\n📊 Displaying interactive 3D visualization...")
    print("💡 TIP: Click and drag to rotate, scroll to zoom, hover for details\\n")
    fig.show()

    return fig
'''

# Updated static matplotlib function
static_function = '''def create_static_3d_spherical_zone(
    performance_data, save_path="outputs/visualizations/3d_spherical_zone.png"
):
    """
    Create static matplotlib 3D visualization following research best practices.

    Shows models in normalized metric space without arbitrary quality zones.

    Args:
        performance_data: Dict with model names as keys and metric dicts as values
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('#FAFAFA')

    # Modern color palette
    colors = {
        "VAE": "#5470C6",
        "GAN": "#EE6666",
        "cGAN": "#91CC75",
        "DDPM": "#FAC858"
    }

    # Plot ideal performance point
    ax.scatter(
        1, 1, 1,
        c='gold',
        s=700,
        alpha=1,
        edgecolors='#2C3E50',
        linewidth=4,
        marker='*',
        label='Ideal Performance',
        zorder=100,
        depthshade=False
    )
    ax.text(1, 1, 1.10, '★ 1.0', fontsize=17, weight='bold', ha='center', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gold', alpha=0.9, linewidth=2))

    # Plot each model
    for model_name, metrics in performance_data.items():
        metrics_list = list(metrics.values())

        if len(metrics_list) >= 3:
            x = metrics_list[0]  # Image Quality
            y = metrics_list[1]  # Training Stability
            z = metrics_list[2]  # Controllability

            # Calculate metrics for annotation
            distance_to_ideal = np.sqrt((x - 1)**2 + (y - 1)**2 + (z - 1)**2)
            avg_score = (x + y + z) / 3

            # Plot model position
            ax.scatter(
                x, y, z,
                c=colors.get(model_name, "#333333"),
                s=400,
                alpha=0.9,
                edgecolors='#2C3E50',
                linewidth=3.5,
                label=f'{model_name} (avg: {avg_score:.3f})',
                depthshade=False,
                zorder=50
            )

            # Add model label
            ax.text(
                x, y, z + 0.08,
                f'{model_name}\\n{avg_score:.3f}',
                fontsize=12,
                weight='bold',
                ha='center',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=colors.get(model_name), alpha=0.9, linewidth=2.5)
            )

    # Axis labels with clear indication that higher = better
    ax.set_xlabel('Image Quality (Normalized) →\\n[0=worst, 1=best]',
                   fontsize=14, weight='bold', labelpad=20, color='#34495E')
    ax.set_ylabel('Training Stability (Normalized) →\\n[0=worst, 1=best]',
                   fontsize=14, weight='bold', labelpad=20, color='#34495E')
    ax.set_zlabel('Controllability (Normalized) ↑\\n[0=worst, 1=best]',
                   fontsize=14, weight='bold', labelpad=20, color='#34495E')

    # Title following research conventions
    title_text = '3D Performance Space: Quantitative Model Comparison\\n' + \
                 'Normalized Metrics [0,1] | Distance to (1,1,1) = Distance to Ideal'
    ax.set_title(title_text, fontsize=18, weight='bold', pad=35,
                 color='#2C3E50', family='sans-serif')

    # Set limits
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.set_zlim(0, 1.15)

    # Enhanced legend
    legend = ax.legend(
        loc='upper left',
        fontsize=11,
        framealpha=0.95,
        edgecolor='#34495E',
        fancybox=True,
        shadow=True,
        borderpad=1.3,
        labelspacing=1.3,
        title='Models (average score)',
        title_fontsize=12
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(2)

    # Grid styling
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.2, color='#BDC3C7')

    # Pane styling
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('#F8F9FA')
    ax.yaxis.pane.set_facecolor('#F8F9FA')
    ax.zaxis.pane.set_facecolor('#F8F9FA')
    ax.xaxis.pane.set_alpha(0.8)
    ax.yaxis.pane.set_alpha(0.8)
    ax.zaxis.pane.set_alpha(0.8)

    # Tick styling
    ax.tick_params(axis='x', labelsize=10, colors='#2C3E50', pad=8)
    ax.tick_params(axis='y', labelsize=10, colors='#2C3E50', pad=8)
    ax.tick_params(axis='z', labelsize=10, colors='#2C3E50', pad=8)

    # Viewing angle
    ax.view_init(elev=22, azim=-58)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    print(f'✅ Static 3D visualization saved to: {save_path}')
'''

# Find and replace the functions in the notebook
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])

        # Replace interactive function
        if "def create_interactive_3d_spherical_zone_colab(" in source:
            print("Updating interactive 3D visualization function...")
            cell["source"] = plotly_function.split("\n")
            # Add newlines to each line for proper formatting
            cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [
                cell["source"][-1]
            ]

        # Replace static function
        if (
            "def create_static_3d_spherical_zone(" in source
            and "Create static matplotlib" in source
        ):
            print("Updating static 3D visualization function...")
            cell["source"] = static_function.split("\n")
            cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [
                cell["source"][-1]
            ]

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✅ Notebook updated successfully!")
print("   - Removed arbitrary quality zones (Basic/Good/Excellent/Elite)")
print("   - Added distance-to-ideal gradient visualization")
print("   - Updated titles and labels to research standards")
print("   - Maintained quantitative metric display")
