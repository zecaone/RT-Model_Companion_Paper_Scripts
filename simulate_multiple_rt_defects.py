#!/usr/bin/env python3
"""
Multi-Defect CMB Simulation - RT Model
Generates CMB map with multiple RT network defects and analyzes resulting
quadrupole (l=2) and octupole (l=3) axis alignment.

This demonstrates how multiple anisotropic defects in the RT network can
produce complex multipole structures and axis misalignments, similar to
the observed "Axis of Evil" anomaly.

Output:
-------
- multi_defect_map.png          : Full CMB map with all defects
- multi_defect_quadrupol.png    : Quadrupole component (l=2)
- multi_defect_octupol.png      : Octupole component (l=3)
- multi_defect_axes.png         : Axis comparison visualization

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================================
# CONFIGURATION
# ============================================================================

# HEALPix parameters
NSIDE = 64               # HEALPix resolution
LMAX = 3                 # Maximum multipole for simulation

# Defect definitions (galactic coordinates)
# Each defect: (longitude, latitude, radius, type)
# Types: 'damping' = amplitude suppression, 'phase' = phase flip
DEFECTS = [
    {
        'lon': 30.0,      # degrees
        'lat': 40.0,      # degrees
        'radius': 10.0,   # degrees
        'type': 'damping',
        'strength': 0.5   # 50% damping
    },
    {
        'lon': 200.0,
        'lat': 10.0,
        'radius': 10.0,
        'type': 'phase',
        'strength': -1.0  # phase flip
    },
    {
        'lon': 330.0,
        'lat': -20.0,
        'radius': 10.0,
        'type': 'damping',
        'strength': 0.5
    }
]

# Visualization parameters
DPI = 300
COLORMAP = 'RdBu_r'      # Red-Blue reversed

# Random seed for reproducibility
RANDOM_SEED = 42

# Output files
OUTPUT_FILES = {
    'full_map': 'multi_defect_map.png',
    'quadrupole': 'multi_defect_quadrupol.png',
    'octupole': 'multi_defect_octupol.png',
    'axes': 'multi_defect_axes.png'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_synthetic_cmb(nside, lmax, seed=RANDOM_SEED):
    """
    Generate synthetic CMB map with low multipoles (l ≤ lmax).
    
    Parameters:
    -----------
    nside : int
        HEALPix NSIDE parameter
    lmax : int
        Maximum multipole
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    cmb_map : array
        Temperature map in μK
    alm : array
        Spherical harmonic coefficients
    """
    np.random.seed(seed)
    
    # Create power spectrum (flat for simplicity)
    cl = np.ones(lmax + 1)
    
    # Generate alm coefficients
    alm = hp.synalm(cl, lmax=lmax)
    
    # Convert to map
    cmb_map = hp.alm2map(alm, nside=nside, lmax=lmax, verbose=False)
    
    return cmb_map, alm


def apply_defect(cmb_map, nside, defect_params):
    """
    Apply RT network defect to CMB map.
    
    Parameters:
    -----------
    cmb_map : array
        Input temperature map
    nside : int
        HEALPix NSIDE
    defect_params : dict
        Defect parameters (lon, lat, radius, type, strength)
        
    Returns:
    --------
    modified_map : array
        Map with defect applied
    affected_pixels : array
        Indices of affected pixels
    """
    modified_map = cmb_map.copy()
    
    # Convert galactic coordinates to colatitude/longitude
    lon_rad = np.radians(defect_params['lon'])
    lat_rad = np.radians(defect_params['lat'])
    theta = np.radians(90.0 - defect_params['lat'])  # colatitude
    phi = lon_rad
    
    # Convert to unit vector
    vec = hp.ang2vec(theta, phi)
    
    # Find pixels within defect radius
    radius_rad = np.radians(defect_params['radius'])
    affected_pixels = hp.query_disc(nside, vec, radius_rad)
    
    # Apply defect modification
    if defect_params['type'] == 'damping':
        # Amplitude damping
        modified_map[affected_pixels] *= defect_params['strength']
    elif defect_params['type'] == 'phase':
        # Phase shift (simple flip as proxy)
        modified_map[affected_pixels] *= defect_params['strength']
    else:
        raise ValueError(f"Unknown defect type: {defect_params['type']}")
    
    return modified_map, affected_pixels


def extract_multipole_component(cmb_map, nside, ell, lmax):
    """
    Extract specific multipole component (l=ell) from map.
    
    Parameters:
    -----------
    cmb_map : array
        Full temperature map
    nside : int
        HEALPix NSIDE
    ell : int
        Multipole to extract
    lmax : int
        Maximum multipole for alm computation
        
    Returns:
    --------
    multipole_map : array
        Map containing only l=ell component
    """
    # Convert map to alm
    alm = hp.map2alm(cmb_map, lmax=lmax)
    
    # Create alm with only l=ell
    alm_filtered = np.zeros_like(alm)
    
    for m in range(-ell, ell + 1):
        idx = hp.Alm.getidx(lmax, ell, abs(m))
        alm_filtered[idx] = alm[idx]
    
    # Convert back to map
    multipole_map = hp.alm2map(alm_filtered, nside=nside, lmax=lmax, verbose=False)
    
    return multipole_map


def find_multipole_axis(multipole_map, nside):
    """
    Find dominant axis direction of multipole component.
    
    Parameters:
    -----------
    multipole_map : array
        Map of specific multipole
    nside : int
        HEALPix NSIDE
        
    Returns:
    --------
    coord : SkyCoord
        Galactic coordinates of axis
    theta, phi : float
        Angular coordinates (radians)
    """
    # Find pixel with maximum amplitude (absolute value)
    ipix_max = np.argmax(np.abs(multipole_map))
    
    # Convert pixel index directly to angles
    # hp.pix2ang returns (theta, phi) for HEALPix RING scheme
    theta, phi = hp.pix2ang(nside, ipix_max)
    
    # Ensure scalar values (healpy sometimes returns 0-d arrays)
    theta = float(theta)
    phi = float(phi)
    
    # Convert to galactic coordinates
    # theta is colatitude (0 at north pole), phi is longitude
    l_deg = np.degrees(phi)
    b_deg = 90.0 - np.degrees(theta)
    
    coord = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
    
    return coord, theta, phi


def plot_mollweide_map(cmb_map, title, filename, unit='μK'):
    """
    Plot HEALPix map in Mollweide projection.
    
    Parameters:
    -----------
    cmb_map : array
        Temperature map
    title : str
        Plot title
    filename : str
        Output filename
    unit : str
        Temperature unit for colorbar
    """
    plt.figure(figsize=(12, 8))
    
    hp.mollview(
        cmb_map,
        title=title,
        unit=unit,
        cmap=COLORMAP,
        hold=False
    )
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_axis_comparison(cmb_map, coord_l2, coord_l3, theta2, phi2, 
                         theta3, phi3, angle, filename):
    """
    Plot CMB map with both multipole axes overlaid.
    
    Parameters:
    -----------
    cmb_map : array
        Base temperature map
    coord_l2, coord_l3 : SkyCoord
        Quadrupole and octupole axis coordinates
    theta2, phi2 : float
        Quadrupole axis angles
    theta3, phi3 : float
        Octupole axis angles
    angle : float
        Separation angle in degrees
    filename : str
        Output filename
    """
    plt.figure(figsize=(12, 8))
    
    hp.mollview(
        cmb_map,
        title=f"Multipole axis comparison (separation: {angle:.1f}°)",
        unit='μK',
        cmap=COLORMAP,
        hold=True
    )
    
    # Plot quadrupole axis (red)
    hp.projplot(
        theta2, phi2, 
        'rx', 
        markersize=12, 
        markeredgewidth=2,
        label=f'Quadrupole (l=2): l={coord_l2.l.deg:.1f}°, b={coord_l2.b.deg:.1f}°'
    )
    
    # Plot octupole axis (blue)
    hp.projplot(
        theta3, phi3, 
        'bx', 
        markersize=12, 
        markeredgewidth=2,
        label=f'Octupole (l=3): l={coord_l3.l.deg:.1f}°, b={coord_l3.b.deg:.1f}°'
    )
    
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    """Generate multi-defect CMB simulation."""
    
    print("=" * 70)
    print("MULTI-DEFECT CMB SIMULATION - RT MODEL")
    print("Analyzing multipole axis alignment with multiple defects")
    print("=" * 70)
    
    npix = hp.nside2npix(NSIDE)
    
    print(f"\nConfiguration:")
    print(f"  HEALPix NSIDE:     {NSIDE}")
    print(f"  Number of pixels:  {npix}")
    print(f"  Max multipole:     l={LMAX}")
    print(f"  Number of defects: {len(DEFECTS)}")
    
    # ========================================================================
    # Step 1: Generate synthetic CMB
    # ========================================================================
    print("\n[1/6] Generating synthetic CMB map...")
    cmb_map, alm_original = generate_synthetic_cmb(NSIDE, LMAX)
    print(f"      Temperature range: [{cmb_map.min():.2e}, {cmb_map.max():.2e}] μK")
    
    # ========================================================================
    # Step 2: Apply defects
    # ========================================================================
    print("\n[2/6] Applying RT network defects...")
    defected_map = cmb_map.copy()
    
    for i, defect in enumerate(DEFECTS):
        print(f"      Defect {i+1}: {defect['type']} at "
              f"(l={defect['lon']:.1f}°, b={defect['lat']:.1f}°), "
              f"r={defect['radius']:.1f}°")
        
        defected_map, affected_pix = apply_defect(defected_map, NSIDE, defect)
        print(f"                Affected {len(affected_pix)} pixels "
              f"({100*len(affected_pix)/npix:.2f}%)")
    
    # Save full defected map
    print("\n[3/6] Saving full defected CMB map...")
    plot_mollweide_map(
        defected_map,
        "CMB temperature map with multiple RT network defects",
        OUTPUT_FILES['full_map']
    )
    
    # ========================================================================
    # Step 3: Extract quadrupole (l=2)
    # ========================================================================
    print("\n[4/6] Extracting quadrupole component (l=2)...")
    map_l2 = extract_multipole_component(defected_map, NSIDE, ell=2, lmax=LMAX)
    print(f"      Quadrupole amplitude range: [{map_l2.min():.2e}, {map_l2.max():.2e}]")
    
    plot_mollweide_map(
        map_l2,
        "Quadrupole component (l=2)",
        OUTPUT_FILES['quadrupole']
    )
    
    # ========================================================================
    # Step 4: Extract octupole (l=3)
    # ========================================================================
    print("\n[5/6] Extracting octupole component (l=3)...")
    map_l3 = extract_multipole_component(defected_map, NSIDE, ell=3, lmax=LMAX)
    print(f"      Octupole amplitude range: [{map_l3.min():.2e}, {map_l3.max():.2e}]")
    
    plot_mollweide_map(
        map_l3,
        "Octupole component (l=3)",
        OUTPUT_FILES['octupole']
    )
    
    # ========================================================================
    # Step 5: Determine axes and separation
    # ========================================================================
    print("\n[6/6] Analyzing multipole axes...")
    
    # Find quadrupole axis
    coord_l2, theta2, phi2 = find_multipole_axis(map_l2, NSIDE)
    print(f"      Quadrupole axis: l={coord_l2.l.deg:.2f}°, b={coord_l2.b.deg:.2f}°")
    
    # Find octupole axis
    coord_l3, theta3, phi3 = find_multipole_axis(map_l3, NSIDE)
    print(f"      Octupole axis:   l={coord_l3.l.deg:.2f}°, b={coord_l3.b.deg:.2f}°")
    
    # Calculate separation angle
    separation_angle = coord_l2.separation(coord_l3).deg
    print(f"      Separation angle: {separation_angle:.2f}°")
    
    # Plot axis comparison
    print("\n      Creating axis comparison plot...")
    plot_axis_comparison(
        defected_map, 
        coord_l2, coord_l3, 
        theta2, phi2, theta3, phi3,
        separation_angle,
        OUTPUT_FILES['axes']
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓ SIMULATION COMPLETE")
    print("=" * 70)
    
    print("\nResults:")
    print(f"  Quadrupole axis (l=2): l = {coord_l2.l.deg:.2f}°, b = {coord_l2.b.deg:.2f}°")
    print(f"  Octupole axis (l=3):   l = {coord_l3.l.deg:.2f}°, b = {coord_l3.b.deg:.2f}°")
    print(f"  Axis separation:       {separation_angle:.2f}°")
    
    print("\nOutput files:")
    for key, filename in OUTPUT_FILES.items():
        print(f"  - {filename}")
    
    print("\nInterpretation:")
    if separation_angle < 30:
        print("  → Strong axis alignment (aligned defects)")
    elif separation_angle < 90:
        print("  → Moderate axis misalignment (complex defect structure)")
    else:
        print("  → Large axis separation (competing defect sources)")
    
    print()


if __name__ == "__main__":
    main()