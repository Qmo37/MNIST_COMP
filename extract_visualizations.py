#!/usr/bin/env python3
"""
Extract visualization functions from Jupyter notebook into a standalone Python file.
"""

import json
import re

# Read the notebook
with open("MNIST_Generative_Models_Complete.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Extract visualization-related cells
viz_code_sections = []
in_viz_section = False

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])

        # Check if this cell contains visualization functions
        if any(
            keyword in source
            for keyword in [
                "def create_radar_chart",
                "def create_interactive_3d",
                "def create_static_3d",
                "def create_heatmap",
                "def create_bar_comparison",
                "def create_training_curves",
                "def create_performance_summary",
                "def create_all_visualizations",
            ]
        ):
            viz_code_sections.append(source)
            print(f"Found visualization cell ({len(source)} chars)")

# Create the standalone file
output_file = "visualizations.py"

header = '''"""
MNIST Generative Models - Visualization Module
==============================================

Standalone Python module containing all visualization functions for comparing
generative models (VAE, GAN, cGAN, DDPM) following research best practices.

This module can be imported and used independently of the Jupyter notebook.

Usage:
    from visualizations import create_all_visualizations

    create_all_visualizations(performance_data, timing_data, losses_dict)

Author: Extracted from MNIST_Generative_Models_Complete.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Dict, List, Optional

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Interactive 3D visualizations will be skipped.")
    print("   Install with: pip install plotly")

# Create output directory
os.makedirs("outputs/visualizations", exist_ok=True)

'''

# Combine all sections
full_content = header + "\n\n" + "\n\n".join(viz_code_sections)

# Write to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_content)

print(f"\n✅ Created {output_file}")
print(f"   Total size: {len(full_content)} characters")
print(f"   Sections extracted: {len(viz_code_sections)}")
