"""
IMMEDIATE FIX for SymPy Compatibility Issue
===========================================
Copy and paste this entire code into a NEW CELL at the TOP of your Colab notebook.
Run it FIRST before any other cells.
"""

# ============================================================================
# STEP 1: Fix SymPy Version
# ============================================================================
print("=" * 70)
print("FIXING SYMPY COMPATIBILITY ISSUE")
print("=" * 70)
print("\nDiagnosing the problem...\n")

import sys
import subprocess

# Check current sympy version
try:
    import sympy

    print(f"Current SymPy version: {sympy.__version__}")

    # Test if it has the 'core' attribute
    if hasattr(sympy, "core"):
        print("✓ SymPy.core exists - compatible!")
    else:
        print("✗ SymPy.core missing - INCOMPATIBLE with PyTorch 2.x")
        print("\nUpgrading SymPy now...")

        # Upgrade sympy
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "sympy>=1.12", "-q"]
        )

        print("✓ SymPy upgraded successfully!")
        print("\n" + "=" * 70)
        print("⚠️  CRITICAL: YOU MUST RESTART THE RUNTIME NOW!")
        print("=" * 70)
        print("\nIn Google Colab:")
        print("  1. Click: Runtime → Restart runtime")
        print("  2. When asked 'Restart runtime?', click YES")
        print("  3. After restart, click: Runtime → Run all")
        print("\n" + "=" * 70)

except ImportError:
    print("SymPy not installed yet - will be installed with other packages")

print("\n" + "=" * 70)
print("FIX APPLIED")
print("=" * 70)
