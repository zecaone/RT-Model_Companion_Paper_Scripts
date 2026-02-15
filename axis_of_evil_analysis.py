#!/usr/bin/env python3
"""
Axis of Evil - Single Directed RT Defect Simulation
Demonstrates perfect axis alignment when RT network has a single
axially symmetric defect structure (m=0 dominated modes).

This represents the idealized case where quadrupole (l=2) and octupole (l=3)
axes are perfectly aligned, contrasting with the more complex multi-defect
scenarios that show axis misalignment.

Output:
-------
- axis_of_evil_quadrupol_map.png    : Quadrupole energy distribution
- axis_of_evil_octupol_map.png      : Octupole energy distribution  
- axis_of_evil_axes_comparison.png  : Combined visualization with both axes

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
NSIDE = 64               # HEALPix resolution (lower for visualization clarity)
LMAX = 10                # Maximum multipole

# Mode amplitudes (axially symmetric - m=0 dominated)
QUADRUPOLE_AMPLITUDE = 1.0   # l=2, m=0
OCTUPOLE_AMPLITUDE = 1.0     # l=3, m=0

# Visualization parameters
DPI = 300
COLORMAP = 'RdBu_r'

# Output files
OUTPUT_FILES = {
    'quadrupole': 'axis_of_evil_quadrupol_map.png',
    'octupole': 'axis_of_evil_octupol_map.png',
    'comparison': 'axis_of_evil_axes_comparison.png'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_axially_symmetric_modes(lmax, l2_amp=1.0, l3_amp=1.0):
    """
    Create alm coefficients with axially symmetric modes (m=0 only).
    
    This represents an RT network with perfect directional symmetry
    along a single axis (z-axis in this case).
    
    Parameters:
    -----------
    lmax : int
        Maximum multipole
    l2_amp : float
        Quadrupole (l=2, m=0) amplitude
    l3_amp : float
        Octupole (l=3, m=0) amplitude
        
    Returns:
    --------
    alm : array
        Spherical harmonic coefficients
    """
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    
    # Set l=2, m=0 (quadrupole)
    idx_l2 = hp.Alm.getidx(lmax, 2, 0)
    alm[idx_l2] = l2_amp + 0.0j
    
    # Set l=3, m=0 (octupole)
    idx_l3 = hp.Alm.getidx(lmax, 3, 0)
    alm[idx_l3] = l3_amp + 0.0j
    
    return alm


def extract_single_multipole(alm, lmax, ell):
    """
    Extract single multipole component from alm array.
    
    Parameters:
    -----------
    alm : array
        Full alm coefficients
    lmax : int
        Maximum multipole
    ell : int
        Multipole to extract
        
    Returns:
    --------
    alm_filtered : array
        alm with only l=ell component
    """
    alm_filtered = np.zeros_like(alm)
    
    # Copy only l=ell coefficients
    for m in range(ell + 1):  # m=0 to m=ell
        idx = hp.Alm.getidx(lmax, ell, m)
        alm_filtered[idx] = alm[idx]
    
    return alm_filtered


def find_energy_maximum(cmb_map, nside):
    """
    Find position of maximum energy (squared amplitude) in map.
    
    Parameters:
    -----------
    cmb_map : array
        Temperature map
    nside : int
        HEALPix NSIDE
        
    Returns:
    --------
    coord : SkyCoord
        Galactic coordinates of maximum
    theta, phi : float
        Angular coordinates (radians)
    """
    # Energy distribution (squared amplitude)
    energy_map = cmb_map**2
    
    # Find pixel with maximum energy
    ipix_max = np.argmax(energy_map)
    
    # Convert to angles
    theta, phi = hp.pix2ang(nside, ipix_max)
    
    # Convert to galactic coordinates
    l_deg = np.degrees(phi)
    b_deg = 90.0 - np.degrees(theta)
    
    coord = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
    
    return coord, theta, phi


def plot_multipole_map(cmb_map, theta_max, phi_max, title, filename, 
                       multipole_label="l=2"):
    """
    Plot multipole map with axis marker.
    
    Parameters:
    -----------
    cmb_map : array
        Temperature map
    theta_max, phi_max : float
        Position of energy maximum (radians)
    title : str
        Plot title
    filename : str
        Output filename
    multipole_label : str
        Label for legend
    """
    plt.figure(figsize=(12, 8))
    
    # Determine color scale
    vmax = np.max(np.abs(cmb_map))
    
    hp.mollview(
        cmb_map,
        title=title,
        unit='Arbitrary units',
        cmap=COLORMAP,
        min=-vmax,
        max=vmax,
        hold=True
    )
    
    # Mark energy maximum (dominant axis)
    hp.projscatter(
        theta_max, phi_max,
        lonlat=False,
        color='red',
        marker='x',
        s=200,
        linewidths=3,
        label=f'{multipole_label} axis'
    )
    
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9)
    
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_combined_comparison(map_l2, map_l3, theta2, phi2, theta3, phi3,
                             coord_l2, coord_l3, separation, filename):
    """
    Plot combined map with both quadrupole and octupole axes.
    
    Parameters:
    -----------
    map_l2, map_l3 : array
        Quadrupole and octupole maps
    theta2, phi2, theta3, phi3 : float
        Axis positions
    coord_l2, coord_l3 : SkyCoord
        Axis coordinates
    separation : float
        Angle between axes (degrees)
    filename : str
        Output filename
    """
    plt.figure(figsize=(14, 9))
    
    # Combined map
    combined = map_l2 + map_l3
    vmax = np.max(np.abs(combined))
    
    hp.mollview(
        combined,
        title=f"Axially Symmetric RT Network: Quadrupole & Octupole\n"
              f"Axis separation: {separation:.2f}° (perfect alignment)",
        unit='Arbitrary units',
        cmap=COLORMAP,
        min=-vmax,
        max=vmax,
        hold=True
    )
    
    # Plot quadrupole axis (red)
    hp.projscatter(
        theta2, phi2,
        lonlat=False,
        color='red',
        marker='x',
        s=200,
        linewidths=3,
        label=f'Quadrupole (l=2): l={coord_l2.l.deg:.2f}°, b={coord_l2.b.deg:.2f}°'
    )
    
    # Plot octupole axis (blue)
    hp.projscatter(
        theta3, phi3,
        lonlat=False,
        color='blue',
        marker='x',
        s=200,
        linewidths=3,
        label=f'Octupole (l=3): l={coord_l3.l.deg:.2f}°, b={coord_l3.b.deg:.2f}°'
    )
    
    plt.legend(loc='lower left', fontsize=11, framealpha=0.95, edgecolor='black')
    
    # Annotation
    textstr = (
        f'Axis Separation: {separation:.2f}°\n'
        f'(Perfect alignment for m=0 dominated modes)'
    )
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
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
# MAIN SIMULATION
# ============================================================================

def main():
    """Generate axially symmetric CMB modes and visualize axis alignment."""
    
    print("=" * 70)
    print("AXIS OF EVIL - SINGLE DIRECTED RT DEFECT")
    print("Perfect axis alignment with axially symmetric modes")
    print("=" * 70)
    
    npix = hp.nside2npix(NSIDE)
    
    print(f"\nConfiguration:")
    print(f"  HEALPix NSIDE:        {NSIDE}")
    print(f"  Number of pixels:     {npix}")
    print(f"  Max multipole:        l={LMAX}")
    print(f"  Mode structure:       Axially symmetric (m=0 only)")
    
    # ========================================================================
    # Step 1: Create axially symmetric modes
    # ========================================================================
    print("\n[1/5] Creating axially symmetric alm coefficients...")
    alm = create_axially_symmetric_modes(
        LMAX, 
        l2_amp=QUADRUPOLE_AMPLITUDE,
        l3_amp=OCTUPOLE_AMPLITUDE
    )
    print(f"  Set l=2, m=0: amplitude = {QUADRUPOLE_AMPLITUDE}")
    print(f"  Set l=3, m=0: amplitude = {OCTUPOLE_AMPLITUDE}")
    
    # ========================================================================
    # Step 2: Extract quadrupole (l=2)
    # ========================================================================
    print("\n[2/5] Extracting quadrupole component (l=2)...")
    alm_l2 = extract_single_multipole(alm, LMAX, ell=2)
    map_l2 = hp.alm2map(alm_l2, NSIDE, lmax=LMAX, verbose=False)
    
    coord_l2, theta2, phi2 = find_energy_maximum(map_l2, NSIDE)
    print(f"  Quadrupole axis: l={coord_l2.l.deg:.2f}°, b={coord_l2.b.deg:.2f}°")
    
    # ========================================================================
    # Step 3: Extract octupole (l=3)
    # ========================================================================
    print("\n[3/5] Extracting octupole component (l=3)...")
    alm_l3 = extract_single_multipole(alm, LMAX, ell=3)
    map_l3 = hp.alm2map(alm_l3, NSIDE, lmax=LMAX, verbose=False)
    
    coord_l3, theta3, phi3 = find_energy_maximum(map_l3, NSIDE)
    print(f"  Octupole axis:   l={coord_l3.l.deg:.2f}°, b={coord_l3.b.deg:.2f}°")
    
    # ========================================================================
    # Step 4: Calculate axis separation
    # ========================================================================
    print("\n[4/5] Calculating axis separation...")
    separation = coord_l2.separation(coord_l3).deg
    print(f"  Axis separation: {separation:.2f}°")
    
    if separation < 1.0:
        print(f"  → Perfect alignment (expected for m=0 dominated modes)")
    else:
        print(f"  → Note: Small deviation due to finite resolution")
    
    # ========================================================================
    # Step 5: Generate visualizations
    # ========================================================================
    print("\n[5/5] Creating visualizations...")
    
    print("  Plotting quadrupole map...")
    plot_multipole_map(
        map_l2, theta2, phi2,
        "Quadrupole Component (l=2)\nAxially Symmetric Mode (m=0)",
        OUTPUT_FILES['quadrupole'],
        multipole_label="l=2"
    )
    
    print("  Plotting octupole map...")
    plot_multipole_map(
        map_l3, theta3, phi3,
        "Octupole Component (l=3)\nAxially Symmetric Mode (m=0)",
        OUTPUT_FILES['octupole'],
        multipole_label="l=3"
    )
    
    print("  Plotting combined comparison...")
    plot_combined_comparison(
        map_l2, map_l3,
        theta2, phi2, theta3, phi3,
        coord_l2, coord_l3,
        separation,
        OUTPUT_FILES['comparison']
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
    print(f"  Axis separation:       {separation:.2f}°")
    
    print("\nOutput files:")
    for key, filename in OUTPUT_FILES.items():
        print(f"  - {filename}")
    
    print("\nInterpretation:")
    print("  This simulation demonstrates the IDEALIZED case of perfect")
    print("  axis alignment when the RT network has a single axially")
    print("  symmetric defect (m=0 dominated modes).")
    print()
    print("  In contrast, the multi-defect simulation shows how multiple")
    print("  overlapping anisotropies lead to axis misalignment, which")
    print("  better matches the observed 'Axis of Evil' in real CMB data.")
    print()


if __name__ == "__main__":
    main()
