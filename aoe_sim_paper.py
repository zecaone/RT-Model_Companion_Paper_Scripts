#!/usr/bin/env python3
"""
Axis of Evil Simulation - RT Model
Simulates anisotropic CMB temperature pattern with preferred axis alignment.

This script generates a 2D visualization of temperature fluctuations with
directional coupling along a preferred axis, representing RT network anisotropy.

Output:
-------
axis_of_evil_simulated_map.png

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Grid parameters
GRID_SIZE = 200          # Grid resolution (NxN)
GRID_EXTENT = 100        # Coordinate range [-GRID_EXTENT, +GRID_EXTENT]

# Anisotropy parameters
PREFERRED_AXIS_ANGLE = 45.0   # degrees (from x-axis)
ELONGATION_RATIO = 5.0        # Major/minor axis ratio
SIGMA_MAJOR = 30.0            # Standard deviation along major axis
SIGMA_MINOR = SIGMA_MAJOR / ELONGATION_RATIO  # Along minor axis

# Noise parameters
NOISE_AMPLITUDE = 0.05   # Relative noise level

# Plot parameters
FIGSIZE = (8, 8)
DPI = 300
COLORMAP = 'RdBu_r'      # Red-Blue reversed (cold=blue, hot=red)

# Output
OUTPUT_FILE = "axis_of_evil_simulated_map.png"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_anisotropic_pattern(grid_size, extent, axis_angle, 
                               sigma_major, sigma_minor, noise_amp):
    """
    Create anisotropic Gaussian temperature pattern with preferred axis.
    
    Parameters:
    -----------
    grid_size : int
        Number of grid points in each direction
    extent : float
        Coordinate range [-extent, +extent]
    axis_angle : float
        Preferred axis angle in degrees
    sigma_major : float
        Gaussian width along major axis
    sigma_minor : float
        Gaussian width along minor axis
    noise_amp : float
        Amplitude of added Gaussian noise
        
    Returns:
    --------
    X, Y : arrays
        Coordinate meshgrids
    Z : array
        Temperature field
    """
    # Create coordinate grid
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Rotation matrix for preferred axis
    theta = np.deg2rad(axis_angle)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Anisotropic Gaussian
    Z = np.exp(-0.5 * ((X_rot / sigma_major) ** 2 + 
                       (Y_rot / sigma_minor) ** 2))
    
    # Add noise
    if noise_amp > 0:
        np.random.seed(RANDOM_SEED)
        Z += noise_amp * np.random.randn(*Z.shape)
    
    return X, Y, Z


def plot_anisotropic_map(X, Y, Z, axis_angle, filename):
    """
    Plot anisotropic temperature map with preferred axis visualization.
    
    Parameters:
    -----------
    X, Y : arrays
        Coordinate meshgrids
    Z : array
        Temperature field
    axis_angle : float
        Preferred axis angle in degrees
    filename : str
        Output filename
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Get coordinate extent
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    
    # Plot temperature field
    im = ax.imshow(
        Z, 
        extent=extent,
        origin='lower', 
        cmap=COLORMAP, 
        norm=Normalize(vmin=0, vmax=Z.max()),
        aspect='equal'
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Relative temperature amplitude $\Delta T / T$", 
                   fontsize=12)
    
    # Axis labels
    ax.set_xlabel("x-direction [arbitrary units]", fontsize=12)
    ax.set_ylabel("y-direction [arbitrary units]", fontsize=12)
    ax.set_title("Simulated anisotropic temperature pattern\n"
                 "RT network with preferred axis coupling", 
                 fontsize=14, pad=15)
    
    # Draw preferred axis line
    ax.plot(
        [X.min(), X.max()],
        [Y.min(), Y.max()],
        color='black', 
        linestyle='--', 
        linewidth=1.5,
        alpha=0.7,
        label='Preferred axis'
    )
    
    # Axis label
    # Position text along the diagonal
    text_x = X.min() + 0.75 * (X.max() - X.min())
    text_y = Y.min() + 0.75 * (Y.max() - Y.min())
    
    ax.text(
        text_x, text_y, 
        f"Preferred axis ({axis_angle:.0f}°)",
        rotation=axis_angle, 
        fontsize=11, 
        color='black',
        ha='center', 
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                  alpha=0.8, edgecolor='none')
    )
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate anisotropic CMB pattern with preferred axis."""
    
    print("=" * 70)
    print("AXIS OF EVIL SIMULATION - RT MODEL")
    print("Generating anisotropic temperature pattern")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Grid size:        {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Preferred axis:   {PREFERRED_AXIS_ANGLE}°")
    print(f"  Elongation ratio: {ELONGATION_RATIO:.1f}")
    print(f"  Noise amplitude:  {NOISE_AMPLITUDE:.3f}")
    
    # Create anisotropic pattern
    print("\n[1/2] Generating anisotropic temperature field...")
    X, Y, Z = create_anisotropic_pattern(
        GRID_SIZE, GRID_EXTENT, PREFERRED_AXIS_ANGLE,
        SIGMA_MAJOR, SIGMA_MINOR, NOISE_AMPLITUDE
    )
    
    print(f"      Temperature range: [{Z.min():.3f}, {Z.max():.3f}]")
    print(f"      Mean: {Z.mean():.3f}, Std: {Z.std():.3f}")
    
    # Plot
    print("\n[2/2] Creating visualization...")
    plot_anisotropic_map(X, Y, Z, PREFERRED_AXIS_ANGLE, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("✓ SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()
