#!/usr/bin/env python3
"""
Cold Spot Visualization Suite - RT Model
Creates comprehensive visualization set for Cold Spot analysis including
original synthetic CMB, damped region, difference map, and statistical
comparison via histogram.

These visualizations complement the variant analysis (A-D) by showing
the direct effect of RT network damping on the temperature field.

Output:
-------
- cold_spot_original_map.png       : Synthetic CMB without defect
- cold_spot_damped_map.png         : With 30% damping at r<10°
- cold_spot_difference_map.png     : Difference map (damped - original)
- cold_spot_histogram.png          : Temperature distribution comparison

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

# HEALPix parameters
NSIDE = 256          # HEALPix resolution
LMAX = 10            # Maximum multipole

# Cold Spot location (galactic coordinates)
COLDSPOT_L = 209.0   # degrees
COLDSPOT_B = -57.0   # degrees

# Defect parameters
DEFECT_RADIUS = 10.0     # degrees (larger than variant radius for broader view)
DAMPING_ALPHA = 0.3      # 30% damping (consistent with variant B)

# Visualization parameters
DPI = 300
COLORMAP = 'RdBu_r'      # Red-Blue reversed (cold=blue, hot=red)
FIGSIZE_MAP = (10, 6)
FIGSIZE_HIST = (10, 7)

# Output directory
OUTPUT_DIR = "figures"

# Output files
OUTPUT_FILES = {
    'original': f'{OUTPUT_DIR}/cold_spot_original_map.png',
    'damped': f'{OUTPUT_DIR}/cold_spot_damped_map.png',
    'difference': f'{OUTPUT_DIR}/cold_spot_difference_map.png',
    'histogram': f'{OUTPUT_DIR}/cold_spot_histogram.png'
}

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_base_modes(lmax=LMAX):
    """
    Generate base temperature map from Y_5^0 + Y_5^±2 modes.
    
    This creates a synthetic CMB pattern with specific multipole structure
    that resembles the Cold Spot's characteristic spatial scales.
    
    Returns:
    --------
    alm : array
        Spherical harmonic coefficients
    """
    # Initialize alm array
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    
    # Set Y_5^0 component (m=0) - axially symmetric
    idx_50 = hp.Alm.getidx(lmax, 5, 0)
    alm[idx_50] = 1.0 + 0.0j
    
    # Set Y_5^2 component (positive m)
    idx_52 = hp.Alm.getidx(lmax, 5, 2)
    alm[idx_52] = 0.5 + 0.0j
    
    # Note: Y_5^-2 is implicitly set via conjugate symmetry
    
    return alm


def galactic_to_vec(lon, lat):
    """
    Convert galactic coordinates (l, b) to HEALPix unit vector.
    
    Parameters:
    -----------
    lon : float
        Galactic longitude in degrees
    lat : float
        Galactic latitude in degrees
        
    Returns:
    --------
    vec : array
        Unit vector [x, y, z]
    """
    theta = np.radians(90.0 - lat)  # colatitude
    phi = np.radians(lon)
    return hp.ang2vec(theta, phi)


def create_defect_mask(nside, center_l, center_b, radius):
    """
    Create boolean mask for defect region on HEALPix map.
    
    Parameters:
    -----------
    nside : int
        HEALPix NSIDE parameter
    center_l, center_b : float
        Center coordinates in galactic degrees
    radius : float
        Defect radius in degrees
        
    Returns:
    --------
    mask : array
        Boolean array (True inside defect, False outside)
    """
    npix = hp.nside2npix(nside)
    center_vec = galactic_to_vec(center_l, center_b)
    
    # Get all pixels within radius
    defect_pixels = hp.query_disc(nside, center_vec, np.radians(radius))
    
    # Create mask
    mask = np.zeros(npix, dtype=bool)
    mask[defect_pixels] = True
    
    return mask


def apply_damping(map_data, mask, alpha):
    """
    Apply damping to temperature map inside defect region.
    
    Parameters:
    -----------
    map_data : array
        Temperature map
    mask : array
        Boolean defect mask
    alpha : float
        Damping factor (0 = no damping, 1 = full suppression)
        
    Returns:
    --------
    damped_map : array
        Map with damping applied
    """
    damped_map = map_data.copy()
    damped_map[mask] *= (1.0 - alpha)
    return damped_map


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_mollweide_map(map_data, title, filename, mark_coldspot=True, 
                       vmin=None, vmax=None):
    """
    Plot HEALPix map in Mollweide projection.
    
    Parameters:
    -----------
    map_data : array
        Temperature map
    title : str
        Plot title
    filename : str
        Output filename
    mark_coldspot : bool
        Whether to mark Cold Spot center
    vmin, vmax : float, optional
        Color scale limits
    """
    fig = plt.figure(figsize=FIGSIZE_MAP)
    
    # Symmetric color scale if not provided
    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(map_data))
        vmin = -abs_max
        vmax = abs_max
    
    hp.mollview(
        map_data,
        title=title,
        cmap=COLORMAP,
        min=vmin,
        max=vmax,
        unit=r'Relative amplitude $\Delta T / T$',
        cbar=True,
        hold=True,
        fig=fig
    )
    
    # Mark Cold Spot center
    if mark_coldspot:
        hp.projplot(
            COLDSPOT_L, COLDSPOT_B,
            'kx',
            markersize=15,
            markeredgewidth=3,
            lonlat=True,
            coord='G',
            label='Cold Spot center'
        )
        
        # Draw defect boundary circle (approximate)
        # Note: Drawing exact circles on Mollweide is complex, 
        # so we just mark the center
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_histogram_comparison(original_map, damped_map, mask, filename):
    """
    Plot histogram comparison of temperature distributions.
    
    Parameters:
    -----------
    original_map : array
        Original temperature map
    damped_map : array
        Damped temperature map
    mask : array
        Defect region mask
    filename : str
        Output filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_HIST)
    
    # Define regions
    inside_defect = mask
    outside_defect = ~mask
    
    # ========================================================================
    # Panel 1: Inside defect region
    # ========================================================================
    
    temps_original_inside = original_map[inside_defect]
    temps_damped_inside = damped_map[inside_defect]
    
    # Remove any invalid values
    temps_original_inside = temps_original_inside[
        (temps_original_inside != hp.UNSEEN) & np.isfinite(temps_original_inside)
    ]
    temps_damped_inside = temps_damped_inside[
        (temps_damped_inside != hp.UNSEEN) & np.isfinite(temps_damped_inside)
    ]
    
    # Calculate statistics
    mean_orig_in = np.mean(temps_original_inside)
    std_orig_in = np.std(temps_original_inside)
    mean_damp_in = np.mean(temps_damped_inside)
    std_damp_in = np.std(temps_damped_inside)
    
    # Plot histograms
    bins = np.linspace(
        min(temps_original_inside.min(), temps_damped_inside.min()),
        max(temps_original_inside.max(), temps_damped_inside.max()),
        50
    )
    
    ax1.hist(temps_original_inside, bins=bins, alpha=0.6, 
             color='steelblue', label='Original', density=True)
    ax1.hist(temps_damped_inside, bins=bins, alpha=0.6, 
             color='darkred', label='With damping (α=0.3)', density=True)
    
    # Mark means
    ax1.axvline(mean_orig_in, color='steelblue', linestyle='--', 
                linewidth=2, alpha=0.8)
    ax1.axvline(mean_damp_in, color='darkred', linestyle='--', 
                linewidth=2, alpha=0.8)
    
    ax1.set_xlabel(r'Temperature $\Delta T / T$', fontsize=12)
    ax1.set_ylabel('Probability density', fontsize=12)
    ax1.set_title(f'Temperature distribution inside defect region (r < {DEFECT_RADIUS}°)', 
                  fontsize=13, pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Original: μ={mean_orig_in:.3e}, σ={std_orig_in:.3e}\n'
                  f'Damped:   μ={mean_damp_in:.3e}, σ={std_damp_in:.3e}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # Panel 2: Outside defect region (for comparison)
    # ========================================================================
    
    temps_original_outside = original_map[outside_defect]
    temps_damped_outside = damped_map[outside_defect]
    
    # Remove invalid values
    temps_original_outside = temps_original_outside[
        (temps_original_outside != hp.UNSEEN) & np.isfinite(temps_original_outside)
    ]
    temps_damped_outside = temps_damped_outside[
        (temps_damped_outside != hp.UNSEEN) & np.isfinite(temps_damped_outside)
    ]
    
    # Calculate statistics
    mean_orig_out = np.mean(temps_original_outside)
    std_orig_out = np.std(temps_original_outside)
    mean_damp_out = np.mean(temps_damped_outside)
    std_damp_out = np.std(temps_damped_outside)
    
    # Plot histograms
    bins2 = np.linspace(
        min(temps_original_outside.min(), temps_damped_outside.min()),
        max(temps_original_outside.max(), temps_damped_outside.max()),
        50
    )
    
    ax2.hist(temps_original_outside, bins=bins2, alpha=0.6, 
             color='steelblue', label='Original', density=True)
    ax2.hist(temps_damped_outside, bins=bins2, alpha=0.6, 
             color='darkgreen', label='With damping (unchanged)', density=True)
    
    # Mark means
    ax2.axvline(mean_orig_out, color='steelblue', linestyle='--', 
                linewidth=2, alpha=0.8)
    ax2.axvline(mean_damp_out, color='darkgreen', linestyle='--', 
                linewidth=2, alpha=0.8)
    
    ax2.set_xlabel(r'Temperature $\Delta T / T$', fontsize=12)
    ax2.set_ylabel('Probability density', fontsize=12)
    ax2.set_title(f'Temperature distribution outside defect region (r ≥ {DEFECT_RADIUS}°)', 
                  fontsize=13, pad=10)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text2 = (f'Original: μ={mean_orig_out:.3e}, σ={std_orig_out:.3e}\n'
                   f'Damped:   μ={mean_damp_out:.3e}, σ={std_damp_out:.3e}')
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all Cold Spot visualization maps."""
    
    print("=" * 70)
    print("COLD SPOT VISUALIZATION SUITE - RT MODEL")
    print("Generating comprehensive visualization set")
    print("=" * 70)
    
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n✓ Created output directory: {OUTPUT_DIR}")
    
    print(f"\nConfiguration:")
    print(f"  HEALPix NSIDE:     {NSIDE}")
    print(f"  Number of pixels:  {hp.nside2npix(NSIDE)}")
    print(f"  Cold Spot:         (l={COLDSPOT_L}°, b={COLDSPOT_B}°)")
    print(f"  Defect radius:     {DEFECT_RADIUS}°")
    print(f"  Damping factor:    α={DAMPING_ALPHA} (30%)")
    
    # ========================================================================
    # Step 1: Generate base synthetic CMB
    # ========================================================================
    print("\n[1/5] Generating synthetic CMB with Y_5^0 + Y_5^±2 modes...")
    
    np.random.seed(RANDOM_SEED)
    
    # Generate alm coefficients
    base_alm = generate_base_modes(LMAX)
    
    # Convert to map
    original_map = hp.alm2map(base_alm, NSIDE, lmax=LMAX, verbose=False)
    
    print(f"  Temperature range: [{original_map.min():.3e}, {original_map.max():.3e}]")
    print(f"  Mean: {original_map.mean():.3e}, Std: {original_map.std():.3e}")
    
    # ========================================================================
    # Step 2: Create defect mask
    # ========================================================================
    print("\n[2/5] Creating defect region mask...")
    
    defect_mask = create_defect_mask(NSIDE, COLDSPOT_L, COLDSPOT_B, DEFECT_RADIUS)
    
    n_pixels_in_defect = np.sum(defect_mask)
    total_pixels = hp.nside2npix(NSIDE)
    
    print(f"  Defect covers {n_pixels_in_defect} pixels "
          f"({100*n_pixels_in_defect/total_pixels:.2f}%)")
    
    # ========================================================================
    # Step 3: Apply damping
    # ========================================================================
    print("\n[3/5] Applying RT network damping...")
    
    damped_map = apply_damping(original_map, defect_mask, DAMPING_ALPHA)
    
    # Calculate difference
    difference_map = damped_map - original_map
    
    print(f"  Damped map range: [{damped_map.min():.3e}, {damped_map.max():.3e}]")
    print(f"  Difference range: [{difference_map.min():.3e}, {difference_map.max():.3e}]")
    
    # ========================================================================
    # Step 4: Create map visualizations
    # ========================================================================
    print("\n[4/5] Creating map visualizations...")
    
    # Determine common color scale for original and damped maps
    vmax_common = max(np.abs(original_map.max()), np.abs(original_map.min()))
    
    print("\n  Creating original map...")
    plot_mollweide_map(
        original_map,
        r"Synthetic CMB: $Y_5^0 + Y_5^{\pm 2}$ modes (no defect)",
        OUTPUT_FILES['original'],
        mark_coldspot=True,
        vmin=-vmax_common,
        vmax=vmax_common
    )
    
    print("  Creating damped map...")
    plot_mollweide_map(
        damped_map,
        f"With RT network damping ($\\alpha={DAMPING_ALPHA}$, $r<{DEFECT_RADIUS}°$)",
        OUTPUT_FILES['damped'],
        mark_coldspot=True,
        vmin=-vmax_common,
        vmax=vmax_common
    )
    
    print("  Creating difference map...")
    plot_mollweide_map(
        difference_map,
        "Difference: Damped - Original\n"
        f"(Effect of RT network defect, $\\alpha={DAMPING_ALPHA}$)",
        OUTPUT_FILES['difference'],
        mark_coldspot=True,
        vmin=None,  # Auto-scale for difference
        vmax=None
    )
    
    # ========================================================================
    # Step 5: Create histogram comparison
    # ========================================================================
    print("\n[5/5] Creating histogram comparison...")
    
    plot_histogram_comparison(
        original_map,
        damped_map,
        defect_mask,
        OUTPUT_FILES['histogram']
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓ VISUALIZATION SUITE COMPLETE")
    print("=" * 70)
    
    print("\nGenerated files:")
    for key, filepath in OUTPUT_FILES.items():
        print(f"  ✓ {filepath}")
    
    print("\nVisualization Summary:")
    print(f"  Original map:    Synthetic CMB with Y_5 modes")
    print(f"  Damped map:      30% temperature suppression in defect region")
    print(f"  Difference map:  Isolated effect of RT network defect")
    print(f"  Histogram:       Statistical comparison of distributions")
    
    print(f"\nDefect Characterization:")
    print(f"  Location:        (l={COLDSPOT_L}°, b={COLDSPOT_B}°)")
    print(f"  Spatial extent:  r < {DEFECT_RADIUS}°")
    print(f"  Damping:         α = {DAMPING_ALPHA} (30% amplitude reduction)")
    print(f"  Affected area:   {100*n_pixels_in_defect/total_pixels:.1f}% of sky")
    
    print("\nThese visualizations complement the variant analysis (A-D) by")
    print("showing the direct effect of RT network topology on the CMB")
    print("temperature field structure.")
    print()


if __name__ == "__main__":
    main()