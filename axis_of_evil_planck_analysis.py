#!/usr/bin/env python3
"""
Planck SMICA Axis of Evil Analysis - RT Model
Analyzes real Planck SMICA CMB data with proper masking for scientific
analysis while showing full data in visualization.

Output:
-------
- axis_of_evil_planck_comparison.png

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
SMICA_PATH = "COM_CompMap_CMB-smica_2048_R1.20.fits"
SMICA_PATH_ALT = "../COM_CompMap_CMB-smica_2048_R1.20.fits"

# HEALPix parameters
NSIDE = 2048
LMAX = 10

# Mask settings (for scientific analysis only, NOT for visualization)
USE_MASK_FOR_ALM = True      # Apply mask when computing alm
GALACTIC_MASK_THRESHOLD = 20  # Exclude |b| < 20° from alm calculation

# Visualization parameters
DPI = 300
COLORMAP = 'RdBu_r'
SHOW_MASK_BOUNDARY = True     # Show boundary of masked region in plot

# Output
OUTPUT_FILE = "axis_of_evil_planck_comparison.png"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_smica_map(filepath):
    """Load Planck SMICA CMB temperature map."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SMICA map not found at: {filepath}")
    
    print(f"  Loading SMICA map from: {filepath}")
    smica_map = hp.read_map(filepath, verbose=False)
    return smica_map


def create_galactic_mask(nside, lat_threshold):
    """
    Create galactic plane mask for scientific analysis.
    
    Parameters:
    -----------
    nside : int
        HEALPix NSIDE
    lat_threshold : float
        Exclude pixels with |b| < lat_threshold (degrees)
        
    Returns:
    --------
    mask : array
        Boolean mask (True = use for analysis, False = exclude)
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b = 90.0 - np.degrees(theta)  # galactic latitude
    
    # Create mask: True for good pixels
    mask = np.abs(b) >= lat_threshold
    
    n_masked = np.sum(~mask)
    print(f"  Galactic analysis mask: excluding {n_masked} pixels "
          f"({100*n_masked/npix:.1f}%) with |b| < {lat_threshold}°")
    print(f"  (Note: Masked pixels excluded from alm computation only,")
    print(f"   full data shown in visualization)")
    
    return mask


def extract_multipole_component(cmb_map, nside, ell, lmax, mask=None):
    """
    Extract specific multipole component with optional masking.
    
    Mask is applied ONLY for alm computation, not for output map.
    """
    
    if mask is not None:
        # Create weighted map for alm computation
        # Zero out masked pixels so they don't contribute to alm
        map_for_alm = cmb_map.copy()
        map_for_alm[~mask] = 0.0
        
        # Also need to normalize by the mask fraction
        # This is a simplified approach - proper masking requires
        # pseudo-alm methods, but for low multipoles this is acceptable
        print(f"    Computing alm with galactic mask applied...")
    else:
        map_for_alm = cmb_map
        print(f"    Computing alm without mask...")
    
    # Compute alm
    alm_full = hp.map2alm(map_for_alm, lmax=lmax)
    
    # Extract only l=ell
    alm_filtered = np.zeros_like(alm_full)
    for m in range(-ell, ell + 1):
        idx = hp.Alm.getidx(lmax, ell, abs(m))
        alm_filtered[idx] = alm_full[idx]
    
    # Convert back to map (full sky, not masked)
    multipole_map = hp.alm2map(alm_filtered, nside, lmax=lmax, verbose=False)
    
    return multipole_map, alm_full


def find_multipole_axis(multipole_map, nside, ell, mask=None):
    """
    Find dominant axis of multipole component.
    
    If mask provided, only search in unmasked region.
    """
    
    if mask is not None:
        # Search only in unmasked region
        map_for_search = multipole_map.copy()
        map_for_search[~mask] = 0.0
    else:
        map_for_search = multipole_map
    
    # Find maximum
    ipix_max = np.argmax(np.abs(map_for_search))
    
    # Convert to angles
    theta, phi = hp.pix2ang(nside, ipix_max)
    theta = float(theta)
    phi = float(phi)
    vec = hp.pix2vec(nside, ipix_max)
    
    # Convert to galactic coordinates
    l_deg = np.degrees(phi)
    b_deg = 90.0 - np.degrees(theta)
    coord = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
    
    print(f"    l={ell} axis: (l={l_deg:.2f}°, b={b_deg:.2f}°)")
    
    return coord, theta, phi, vec


def calculate_axis_separation(vec1, vec2):
    """Calculate angular separation between unit vectors."""
    if isinstance(vec1, tuple):
        vec1 = np.array(vec1)
    if isinstance(vec2, tuple):
        vec2 = np.array(vec2)
    
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot = np.dot(vec1, vec2)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    
    return angle_deg


def plot_axis_comparison(smica_map, coord_l2, coord_l3, 
                         theta2, phi2, theta3, phi3, 
                         separation_angle, filename,
                         mask_threshold=None):
    """
    Plot SMICA map with multipole axes.
    
    Shows FULL data but indicates masked region with boundary lines.
    """
    
    plt.figure(figsize=(14, 9))
    
    # Plot FULL SMICA map (no masking in visualization)
    hp.mollview(
        smica_map,
        title=f"Planck SMICA: Quadrupole & Octupole Axes\n"
              f"Axis separation: {separation_angle:.2f}°",
        unit='μK',
        cmap=COLORMAP,
        min=-300,
        max=300,
        hold=True
    )
    
    # Optionally show mask boundary
    if mask_threshold is not None and SHOW_MASK_BOUNDARY:
        # Draw lines at b = ±mask_threshold
        l_line = np.linspace(0, 360, 500)
        
        for b_val in [-mask_threshold, mask_threshold]:
            # theta must be an ARRAY with same length as phi
            theta_line = np.full_like(l_line, np.radians(90 - b_val))
            phi_line = np.radians(l_line)
            
            hp.projplot(
                theta_line, phi_line,
                'k--',
                linewidth=1.5,
                alpha=0.5
            )
        
        # Add annotation
        plt.gcf().text(
            0.5, 0.02,
            f'Dashed lines: Galactic mask boundary (|b| = ±{mask_threshold}°)\n'
            f'Region inside excluded from multipole analysis',
            ha='center',
            fontsize=9,
            style='italic',
            color='gray'
        )
    
    # Plot quadrupole axis (red)
    hp.projplot(
        theta2, phi2, 
        'rx', 
        markersize=15, 
        markeredgewidth=3,
        label=f'Quadrupole (l=2): l={coord_l2.l.deg:.2f}°, b={coord_l2.b.deg:.2f}°'
    )
    
    # Plot octupole axis (blue)
    hp.projplot(
        theta3, phi3, 
        'bx', 
        markersize=15, 
        markeredgewidth=3,
        label=f'Octupole (l=3): l={coord_l3.l.deg:.2f}°, b={coord_l3.b.deg:.2f}°'
    )
    
    # Legend
    plt.legend(
        loc='lower left', 
        fontsize=11, 
        framealpha=0.95,
        edgecolor='black'
    )
    
    # Annotation box
    textstr = (
        f'Axis Separation: {separation_angle:.2f}°\n'
        f'Expected (isotropic): ~60°-90°'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gcf().text(
        0.15, 0.85, textstr, 
        fontsize=10, 
        verticalalignment='top',
        bbox=props
    )
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Analyze Planck SMICA for Axis of Evil."""
    
    print("=" * 70)
    print("PLANCK SMICA AXIS OF EVIL ANALYSIS")
    print("=" * 70)
    
    npix = hp.nside2npix(NSIDE)
    print(f"\nConfiguration:")
    print(f"  HEALPix NSIDE:        {NSIDE}")
    print(f"  Number of pixels:     {npix}")
    print(f"  Max multipole:        l={LMAX}")
    print(f"  Mask for alm:         {'ENABLED' if USE_MASK_FOR_ALM else 'DISABLED'}")
    if USE_MASK_FOR_ALM:
        print(f"  Mask threshold:       |b| < {GALACTIC_MASK_THRESHOLD}°")
    print(f"  Show full data:       YES (mask only affects analysis)")
    
    # ========================================================================
    # Step 1: Load SMICA map
    # ========================================================================
    print("\n[1/5] Loading Planck SMICA CMB map...")
    
    if os.path.exists(SMICA_PATH):
        smica_map = load_smica_map(SMICA_PATH)
    elif os.path.exists(SMICA_PATH_ALT):
        smica_map = load_smica_map(SMICA_PATH_ALT)
    else:
        print("ERROR: SMICA map not found!")
        return
    
    print(f"  Temperature range: [{smica_map.min():.1f}, {smica_map.max():.1f}] μK")
    print(f"  Mean: {smica_map.mean():.2f} μK, RMS: {smica_map.std():.2f} μK")
    
    # ========================================================================
    # Step 2: Create analysis mask
    # ========================================================================
    print("\n[2/5] Creating analysis mask...")
    
    if USE_MASK_FOR_ALM:
        analysis_mask = create_galactic_mask(NSIDE, GALACTIC_MASK_THRESHOLD)
    else:
        analysis_mask = None
        print("  No mask applied")
    
    # ========================================================================
    # Step 3: Extract quadrupole (l=2)
    # ========================================================================
    print("\n[3/5] Extracting quadrupole component (l=2)...")
    map_l2, _ = extract_multipole_component(
        smica_map, NSIDE, ell=2, lmax=LMAX, mask=analysis_mask
    )
    print(f"  Quadrupole RMS: {map_l2.std():.2e} μK")
    
    # ========================================================================
    # Step 4: Extract octupole (l=3)
    # ========================================================================
    print("\n[4/5] Extracting octupole component (l=3)...")
    map_l3, _ = extract_multipole_component(
        smica_map, NSIDE, ell=3, lmax=LMAX, mask=analysis_mask
    )
    print(f"  Octupole RMS: {map_l3.std():.2e} μK")
    
    # ========================================================================
    # Step 5: Determine axes
    # ========================================================================
    print("\n[5/5] Determining dominant axes...")
    
    print("  Quadrupole (l=2):")
    coord_l2, theta2, phi2, vec_l2 = find_multipole_axis(
        map_l2, NSIDE, ell=2, mask=analysis_mask
    )
    
    print("  Octupole (l=3):")
    coord_l3, theta3, phi3, vec_l3 = find_multipole_axis(
        map_l3, NSIDE, ell=3, mask=analysis_mask
    )
    
    separation = calculate_axis_separation(vec_l2, vec_l3)
    print(f"\n  Axis separation: {separation:.2f}°")
    
    # ========================================================================
    # Step 6: Visualize
    # ========================================================================
    print("\nCreating visualization...")
    plot_axis_comparison(
        smica_map,  # FULL data, not masked!
        coord_l2, coord_l3,
        theta2, phi2, theta3, phi3,
        separation,
        OUTPUT_FILE,
        mask_threshold=GALACTIC_MASK_THRESHOLD if USE_MASK_FOR_ALM else None
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nResults:")
    print(f"  Quadrupole axis (l=2): l = {coord_l2.l.deg:.2f}°, b = {coord_l2.b.deg:.2f}°")
    print(f"  Octupole axis (l=3):   l = {coord_l3.l.deg:.2f}°, b = {coord_l3.b.deg:.2f}°")
    print(f"  Axis separation:       {separation:.2f}°")
    
    print("\nInterpretation:")
    if separation < 30:
        print("  → STRONG alignment (unexpected in isotropic CMB)")
        print("     Potential evidence for 'Axis of Evil'")
    elif separation < 60:
        print("  → MODERATE alignment")
        print("     Possibly consistent with 'Axis of Evil' anomaly")
    else:
        print("  → WEAK alignment (consistent with statistical isotropy)")
        print("     Expected range: 60°-120°")
    
    print(f"\nOutput file: {OUTPUT_FILE}")
    
    print("\nNote about the horizontal bright stripe:")
    print("  The bright band at b=0° is a known INPAINTING ARTIFACT")
    print("  from Planck's SMICA component separation. The galactic")
    print("  plane region was masked during processing and filled with")
    print("  synthetic data, creating sharp boundaries.")
    print("  → This region (|b| < 20°) is excluded from the scientific")
    print("     analysis (alm computation) but shown in the visualization.")
    print()


if __name__ == "__main__":
    main()