#!/usr/bin/env python3
"""
MNIST Generative Models - Visualization Launcher
===============================================

Easy launcher script to run the local visualization demo from the main directory.
This script handles dependency installation and execution automatically.
"""

import os
import sys
import subprocess
import importlib.util


def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("🎨 MNIST Generative Models - Visualization Launcher")
    print("=" * 55)
    print()

    # Check if we're in the right directory
    if not os.path.exists("local_visualizations"):
        print("❌ Error: 'local_visualizations' directory not found!")
        print("   Please run this script from the MNIST_COMP directory.")
        return 1

    # Check and install required packages
    required_packages = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scipy": "scipy",
    }

    optional_packages = {"plotly": "plotly", "pandas": "pandas", "tqdm": "tqdm"}

    print("🔍 Checking dependencies...")

    # Install required packages
    for import_name, pip_name in required_packages.items():
        if not check_package(import_name):
            print(f"📦 Installing required package: {pip_name}")
            try:
                install_package(pip_name)
                print(f"✅ Successfully installed {pip_name}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {pip_name}")
                print(f"   Please install manually: pip install {pip_name}")
                return 1
        else:
            print(f"✅ {pip_name} already installed")

    # Install optional packages (best effort)
    for import_name, pip_name in optional_packages.items():
        if not check_package(import_name):
            print(f"📦 Installing optional package: {pip_name}")
            try:
                install_package(pip_name)
                print(f"✅ Successfully installed {pip_name}")
            except subprocess.CalledProcessError:
                print(
                    f"⚠️  Optional package {pip_name} failed to install (continuing anyway)"
                )
        else:
            print(f"✅ {pip_name} already installed")

    print()
    print("🚀 Launching visualization demo...")
    print("-" * 40)

    # Change to local_visualizations directory and run the script
    original_dir = os.getcwd()
    try:
        os.chdir("local_visualizations")

        # Import and run the visualization demo
        sys.path.insert(0, ".")
        from visualization_demo import main as demo_main

        demo_main()

    except ImportError as e:
        print(f"❌ Error importing visualization demo: {e}")
        print("   Trying to run as subprocess...")
        try:
            subprocess.run([sys.executable, "visualization_demo.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run visualization demo: {e}")
            return 1

    except Exception as e:
        print(f"❌ Error running visualization demo: {e}")
        return 1

    finally:
        os.chdir(original_dir)

    print()
    print("✨ Visualization demo completed!")
    print(
        f"📁 Check the 'local_visualizations/outputs/' directory for all generated files."
    )
    print()
    print("💡 Key outputs:")
    print("   • local_visualizations/outputs/visualizations/ - All charts and plots")
    print("   • local_visualizations/outputs/images/ - Generated image grids")
    print("   • local_visualizations/outputs/tables/ - Performance summaries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
