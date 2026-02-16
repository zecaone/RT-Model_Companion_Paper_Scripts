#!/usr/bin/env python3
"""
Cold Spot Simulation - RT Model
Generates four variants (A-D) showing effects of RT network defects
on spherical harmonic temperature modulations.

Variants:
A: Y_5^0 + Y_5^±2 without defect (reference)
B: Same modes with damping (α=0.3, r<7°)
C: Phase shift (π/2, r<7°)
D: Damping + phase shift combined

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================================
# CONFIGURATION
# ============================================================================

NSIDE = 256  # HEALPix resolution (higher = better resolution)
LMAX = 10    # Maximum multipole (we need l=5)

# Cold Spot position (galactic coordinates)
COLDSPOT_L = 209.0  # degrees
COLDSPOT_B = -57.0  # degrees

# Defect parameters
DEFECT_RADIUS = 7.0      # degrees
DAMPING_ALPHA = 0.3      # damping factor (0 = no damping, 1 = full damping)
PHASE_SHIFT = np.pi / 2  # radians

# Output settings
DPI = 300
FIGSIZE = (10, 6)
CMAP = 'RdBu_r'  # Red-Blue colormap (reversed)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def galactic_to_vec(l, b):
    """
    Convert galactic coordinates (l, b) in degrees to unit vector.
    
    Parameters:
    -----------
    l : float
        Galactic longitude in degrees
    b : float
        Galactic latitude in degrees
        
    Returns:
    --------
    vec : array
        Unit vector [x, y, z]
    """
    l_rad = np.deg2rad(l)
    b_rad = np.deg2rad(b)
    
    x = np.cos(b_rad) * np.cos(l_rad)
    y = np.cos(b_rad) * np.sin(l_rad)
    z = np.sin(b_rad)
    
    return np.array([x, y, z])


def angular_distance(vec1, vec2):
    """
    Calculate angular distance between two unit vectors in degrees.
    
    Parameters:
    -----------
    vec1, vec2 : array
        Unit vectors
        
    Returns:
    --------
    angle : float
        Angular distance in degrees
    """
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


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
    
    mask = np.zeros(npix, dtype=bool)
    
    for ipix in range(npix):
        pix_vec = np.array(hp.pix2vec(nside, ipix))
        dist = angular_distance(center_vec, pix_vec)
        
        if dist < radius:
            mask[ipix] = True
            
    return mask


def generate_base_modes():
    """
    Generate base temperature map from Y_5^0 + Y_5^±2 modes.
    
    Returns:
    --------
    alm : array
        Spherical harmonic coefficients
    """
    # Initialize alm array
    alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    
    # Set Y_5^0 component (m=0)
    idx_50 = hp.Alm.getidx(LMAX, 5, 0)
    alm[idx_50] = 1.0 + 0.0j
    
    # Set Y_5^2 component (positive m)
    idx_52 = hp.Alm.getidx(LMAX, 5, 2)
    alm[idx_52] = 0.5 + 0.0j
    
    # Set Y_5^-2 component (negative m, must be conjugate due to reality)
    # For healpy: a_lm for m<0 is stored implicitly via conjugate symmetry
    # So we only need to set positive m values
    
    return alm


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


def apply_phase_shift(alm, mask, phase, nside):
    """
    Apply phase shift to alm coefficients in defect region.
    
    Parameters:
    -----------
    alm : array
        Spherical harmonic coefficients
    mask : array
        Boolean defect mask
    phase : float
        Phase shift in radians
    nside : int
        HEALPix NSIDE
        
    Returns:
    --------
    shifted_alm : array
        Modified alm coefficients
    """
    # Convert alm to map
    base_map = hp.alm2map(alm, nside)
    
    # Apply phase shift in defect region
    # Phase shift corresponds to multiplying complex amplitude by e^(i*phase)
    shifted_map = base_map.copy()
    
    # Extract defect region and apply rotation in complex plane
    # For real maps, phase shift manifests as sign flip or amplitude modulation
    shifted_map[mask] *= np.cos(phase)
    
    # Convert back to alm
    shifted_alm = hp.map2alm(shifted_map, lmax=LMAX)
    
    return shifted_alm


def plot_mollweide(map_data, title, filename, vmin=None, vmax=None):
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
    vmin, vmax : float, optional
        Color scale limits
    """
    # Calculate vmin/vmax from data if not provided
    if vmin is None:
        vmin = -np.max(np.abs(map_data))
    if vmax is None:
        vmax = np.max(np.abs(map_data))
    
    fig = plt.figure(figsize=FIGSIZE)
    
    hp.mollview(
        map_data,
        title=title,
        cmap=CMAP,
        min=vmin,
        max=vmax,
        unit=r'$\Delta T / T$',
        cbar=True,
        hold=True,
        fig=fig
    )
    
    # Mark Cold Spot position
    hp.projplot(
        COLDSPOT_L, COLDSPOT_B,
        'kx',
        markersize=15,
        markeredgewidth=2,
        lonlat=True,
        coord='G'
    )
    
    # Add defect circle
    circle_l = np.linspace(0, 360, 100)
    circle_b = np.zeros_like(circle_l)
    
    # Rotate circle to Cold Spot position
    center_vec = galactic_to_vec(COLDSPOT_L, COLDSPOT_B)
    
    # Simplified circle drawing (approximation)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = DEFECT_RADIUS * np.cos(theta)
    circle_y = DEFECT_RADIUS * np.sin(theta)
    
    # Note: Exact circle projection is complex; using marker instead
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    """Generate all four variants."""
    
    print("=" * 70)
    print("COLD SPOT SIMULATION - RT MODEL")
    print("Generating four variants (A-D)")
    print("=" * 70)
    
    # Create defect mask
    print("\n[1/5] Creating defect mask...")
    defect_mask = create_defect_mask(NSIDE, COLDSPOT_L, COLDSPOT_B, DEFECT_RADIUS)
    print(f"      Defect covers {np.sum(defect_mask)} pixels "
          f"({100*np.sum(defect_mask)/len(defect_mask):.2f}%)")
    
    # Generate base alm (Y_5^0 + Y_5^±2)
    print("\n[2/5] Generating base spherical harmonic modes...")
    base_alm = generate_base_modes()
    base_map = hp.alm2map(base_alm, NSIDE)
    print(f"      Base map: min={np.min(base_map):.3e}, max={np.max(base_map):.3e}")
    
    # Determine common color scale for all plots
    vmin = -np.max(np.abs(base_map)) * 1.2
    vmax = np.max(np.abs(base_map)) * 1.2
    
    # ========================================================================
    # VARIANT A: Reference (no defect)
    # ========================================================================
    print("\n[3/5] Variant A: Reference (no defect)...")
    plot_mollweide(
        base_map,
        r"Variant A: $Y_5^0 + Y_5^{\pm 2}$ without defect",
        "cold_spot_variant_A.png",
        vmin=vmin,
        vmax=vmax
    )
    
    # ========================================================================
    # VARIANT B: Damping only
    # ========================================================================
    print("\n[4/5] Variant B: Damping (α=0.3, r<7°)...")
    damped_map = apply_damping(base_map, defect_mask, DAMPING_ALPHA)
    plot_mollweide(
        damped_map,
        r"Variant B: Damping in defect ($\alpha=0.3$, $r<7°$)",
        "cold_spot_variant_B.png",
        vmin=vmin,
        vmax=vmax
    )
    
    # ========================================================================
    # VARIANT C: Phase shift only
    # ========================================================================
    print("\n[5/5] Variant C: Phase shift (π/2, r<7°)...")
    phase_shifted_map = base_map.copy()
    # Apply phase shift as amplitude modulation
    phase_shifted_map[defect_mask] *= np.cos(PHASE_SHIFT)
    
    plot_mollweide(
        phase_shifted_map,
        r"Variant C: Phase shift ($\pi/2$) in defect",
        "cold_spot_variant_C.png",
        vmin=vmin,
        vmax=vmax
    )
    
    # ========================================================================
    # VARIANT D: Combined (damping + phase shift)
    # ========================================================================
    print("\n[6/5] Variant D: Damping + phase shift...")
    combined_map = base_map.copy()
    
    # Apply damping
    combined_map[defect_mask] *= (1.0 - DAMPING_ALPHA)
    
    # Apply phase shift
    combined_map[defect_mask] *= np.cos(PHASE_SHIFT)
    
    plot_mollweide(
        combined_map,
        r"Variant D: Damping ($\alpha=0.3$) + phase shift ($\pi/2$)",
        "cold_spot_variant_D.png",
        vmin=vmin,
        vmax=vmax
    )
    
    print("\n" + "=" * 70)
    print("✓ ALL VARIANTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nOutput files:")
    print("  - cold_spot_variant_A.png")
    print("  - cold_spot_variant_B.png")
    print("  - cold_spot_variant_C.png")
    print("  - cold_spot_variant_D.png")
    print()


if __name__ == "__main__":
    main()