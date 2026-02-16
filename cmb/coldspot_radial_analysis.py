#!/usr/bin/env python3
"""
Cold Spot Statistical Significance Analysis - RT Model
Compares the CMB Cold Spot radial profile against random control regions
to assess statistical significance and anomaly strength.

This analysis is crucial for establishing whether the Cold Spot represents
a genuine cosmological anomaly or just a statistical fluctuation. The
comparison with control regions allows quantification of the deviation
in terms of standard deviations (σ).

In the RT model context, a significant deviation would support the
hypothesis that localized network defects create observable temperature
patterns distinct from standard quantum fluctuations.

Output:
-------
- coldspot_control_comparison.png     : Cold Spot vs control regions
- coldspot_statistical_analysis.png   : Statistical significance analysis
- coldspot_control_profiles.csv       : Control region statistics
- coldspot_significance_report.txt    : Detailed statistical report

Author: Urs Krafzig
Date: 2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
SMICA_PATH = "COM_CompMap_CMB-smica_2048_R1.20.fits"
SMICA_PATH_ALT = "../COM_CompMap_CMB-smica_2048_R1.20.fits"

# HEALPix parameters
NSIDE = 2048

# Cold Spot parameters (galactic coordinates)
COLDSPOT_L = 209.0
COLDSPOT_B = -57.0

# Analysis parameters
MAX_RADIUS = 10.0         # Maximum radius for profile (degrees)
RADIAL_STEP = 0.5         # Radial bin width (degrees)
N_CONTROL_REGIONS = 500   # Number of random control regions
GALACTIC_MASK = 20        # Avoid galactic plane |b| < threshold (degrees)

# Random seed for reproducibility
RANDOM_SEED = 42

# Visualization parameters
DPI = 300
COLORMAP = 'RdBu_r'

# Output files
OUTPUT_FILES = {
    'comparison': 'coldspot_control_comparison.png',
    'significance': 'coldspot_statistical_analysis.png',
    'radial_analysis': 'coldspot_radial_analysis.png',  # NEU!
    'profiles_csv': 'coldspot_control_profiles.csv',
    'report': 'coldspot_significance_report.txt'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_smica_map(filepath):
    """Load Planck SMICA CMB temperature map."""
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SMICA map not found at: {filepath}")
    
    print(f"  Loading SMICA map from: {filepath}")
    smica_map = hp.read_map(filepath, verbose=False)
    return smica_map


def compute_radial_ring_profile(cmb_map, nside, center_l, center_b, 
                                max_radius, step_size):
    """
    Compute radial temperature profile using concentric rings.
    
    Parameters:
    -----------
    cmb_map : array
        Full-sky CMB map
    nside : int
        HEALPix NSIDE
    center_l, center_b : float
        Center coordinates (galactic degrees)
    max_radius : float
        Maximum radius (degrees)
    step_size : float
        Width of radial bins (degrees)
        
    Returns:
    --------
    radii : array
        Radial bin centers (degrees)
    profile_mean : array
        Mean temperature in each ring
    profile_std : array
        Standard deviation in each ring
    """
    # Convert center to vector
    theta = np.radians(90.0 - center_b)
    phi = np.radians(center_l)
    vec = hp.ang2vec(theta, phi)
    
    # Radial bins
    n_bins = int(max_radius / step_size)
    radii = np.linspace(step_size/2, max_radius - step_size/2, n_bins)
    
    profile_mean = []
    profile_std = []
    
    for r in radii:
        # Define ring as annulus between r-step/2 and r+step/2
        inner_radius = np.radians(r - step_size/2)
        outer_radius = np.radians(r + step_size/2)
        
        # Get pixels in inner and outer discs
        if inner_radius > 0:
            inner_pixels = set(hp.query_disc(nside, vec, inner_radius))
        else:
            inner_pixels = set()
        
        outer_pixels = set(hp.query_disc(nside, vec, outer_radius))
        
        # Ring pixels = outer - inner
        ring_pixels = list(outer_pixels - inner_pixels)
        
        # Get temperatures
        temps = cmb_map[ring_pixels]
        temps = temps[temps != hp.UNSEEN]
        
        if len(temps) > 0:
            profile_mean.append(np.mean(temps))
            profile_std.append(np.std(temps))
        else:
            profile_mean.append(np.nan)
            profile_std.append(np.nan)
    
    return radii, np.array(profile_mean), np.array(profile_std)


def generate_control_regions(n_regions, galactic_mask, seed=RANDOM_SEED):
    """
    Generate random control region coordinates avoiding galactic plane.
    
    Parameters:
    -----------
    n_regions : int
        Number of control regions
    galactic_mask : float
        Avoid |b| < galactic_mask (degrees)
    seed : int
        Random seed
        
    Returns:
    --------
    coords : SkyCoord
        Array of galactic coordinates
    """
    np.random.seed(seed)
    
    # Random galactic coordinates
    lon = np.random.uniform(0, 360, n_regions)
    lat = np.random.uniform(-90 + galactic_mask, 90 - galactic_mask, n_regions)
    
    coords = SkyCoord(l=lon*u.deg, b=lat*u.deg, frame='galactic')
    
    return coords


def compute_control_profiles(cmb_map, nside, control_coords, 
                            max_radius, step_size):
    """
    Compute radial profiles for all control regions.
    
    Parameters:
    -----------
    cmb_map : array
        Full-sky CMB map
    nside : int
        HEALPix NSIDE
    control_coords : SkyCoord
        Control region coordinates
    max_radius : float
        Maximum radius (degrees)
    step_size : float
        Radial bin width (degrees)
        
    Returns:
    --------
    radii : array
        Radial bin centers
    profiles : array
        Temperature profiles (shape: n_regions x n_bins)
    mean_profile : array
        Mean profile across all control regions
    std_profile : array
        Standard deviation across control regions
    """
    n_regions = len(control_coords)
    n_bins = int(max_radius / step_size)
    
    profiles = np.zeros((n_regions, n_bins))
    
    print(f"  Computing profiles for {n_regions} control regions...")
    
    for i, coord in enumerate(control_coords):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{n_regions}")
        
        radii, profile_mean, _ = compute_radial_ring_profile(
            cmb_map, nside,
            coord.l.deg, coord.b.deg,
            max_radius, step_size
        )
        
        profiles[i, :] = profile_mean
    
    # Compute statistics across all control regions
    mean_profile = np.nanmean(profiles, axis=0)
    std_profile = np.nanstd(profiles, axis=0)
    
    return radii, profiles, mean_profile, std_profile


def calculate_significance(coldspot_profile, control_mean, control_std):
    """
    Calculate statistical significance of Cold Spot deviation.
    
    Parameters:
    -----------
    coldspot_profile : array
        Cold Spot temperature profile
    control_mean : array
        Mean control region profile
    control_std : array
        Standard deviation of control profiles
        
    Returns:
    --------
    sigma_deviation : array
        Deviation in units of σ at each radius
    max_sigma : float
        Maximum deviation
    max_sigma_radius : float
        Radius of maximum deviation
    """
    # Calculate deviation in units of σ
    sigma_deviation = (coldspot_profile - control_mean) / control_std
    
    # Find maximum deviation
    max_sigma = np.nanmin(sigma_deviation)  # Most negative (coldest)
    max_sigma_idx = np.nanargmin(sigma_deviation)
    
    return sigma_deviation, max_sigma, max_sigma_idx


def plot_comparison(radii, coldspot_mean, coldspot_std,
                   control_mean, control_std, filename):
    """Plot Cold Spot vs control regions comparison."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Control regions (shaded band)
    ax.fill_between(
        radii,
        control_mean - control_std,
        control_mean + control_std,
        color='lightgray',
        alpha=0.7,
        label='Control regions (±1σ)'
    )
    
    # Control mean
    ax.plot(
        radii, control_mean,
        color='gray',
        linestyle='--',
        linewidth=2,
        label='Mean of control regions'
    )
    
    # Cold Spot
    ax.errorbar(
        radii, coldspot_mean,
        yerr=coldspot_std,
        fmt='o-',
        color='steelblue',
        linewidth=2,
        markersize=6,
        capsize=5,
        label='Cold Spot',
        zorder=10
    )
    
    # Zero line
    ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Radial distance from center [°]', fontsize=13)
    ax.set_ylabel('Temperature [μK]', fontsize=13)
    ax.set_title('CMB Cold Spot vs Random Control Regions\n'
                 f'Statistical Comparison ({N_CONTROL_REGIONS} control regions)',
                 fontsize=15, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")

def plot_radial_analysis_combined(radii, coldspot_mean, control_mean, 
                                   control_std, sigma_deviation, filename):
    """
    Create combined 2-panel plot matching paper layout.
    
    Top: Radial temperature profile with ±1σ, ±2σ, ±3σ control bands
    Bottom: Z-score profile with significance thresholds
    """
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), 
                                             sharex=True, 
                                             gridspec_kw={'height_ratios': [1.2, 1]})
    
    # ========================================================================
    # TOP PANEL: Temperature Profile
    # ========================================================================
    
    # Control regions: ±1σ, ±2σ, ±3σ bands
    ax_top.fill_between(radii, 
                        control_mean - 3*control_std,
                        control_mean + 3*control_std,
                        color='lightgray', alpha=0.3, label='±3σ')
    ax_top.fill_between(radii,
                        control_mean - 2*control_std,
                        control_mean + 2*control_std,
                        color='gray', alpha=0.4, label='±2σ')
    ax_top.fill_between(radii,
                        control_mean - control_std,
                        control_mean + control_std,
                        color='darkgray', alpha=0.5, label='±1σ')
    
    # Control mean (dashed line)
    ax_top.plot(radii, control_mean, '--', color='gray', linewidth=2,
                label='Control mean', zorder=5)
    
    # Cold Spot profile (blue solid line)
    ax_top.plot(radii, coldspot_mean, 'o-', color='steelblue', 
                linewidth=2.5, markersize=6, label='Cold Spot', zorder=10)
    
    # Vertical line at r=5° (Cruz et al. characteristic scale)
    ax_top.axvline(5.0, color='black', linestyle=':', linewidth=1.5, 
                   alpha=0.7, label='r = 5° (Cruz et al.)')
    
    # Styling
    ax_top.set_ylabel('Temperature [μK]', fontsize=13)
    ax_top.set_title('Radial temperature profile analysis of the CMB Cold Spot from\n'
                     'Planck 2018 SMICA data', fontsize=14, pad=15)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.95)
    
    # ========================================================================
    # BOTTOM PANEL: Z-score Profile
    # ========================================================================
    
    # Z-score profile
    significant_mask = np.abs(sigma_deviation) > 3
    
    # All points (gray line)
    ax_bottom.plot(radii, sigma_deviation, '-', color='gray', 
                   linewidth=1.5, alpha=0.7, zorder=1)
    
    # Significant points (red scatter)
    if np.any(significant_mask):
        ax_bottom.scatter(radii[significant_mask], 
                         sigma_deviation[significant_mask],
                         c='red', s=80, marker='o', zorder=10,
                         label=f'Significant (|Z| > 3σ)')
    
    # Threshold lines
    ax_bottom.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_bottom.axhline(3, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label='±3σ threshold')
    ax_bottom.axhline(-3, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Vertical line at r=5°
    ax_bottom.axvline(5.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Styling
    ax_bottom.set_xlabel('Radial distance [°]', fontsize=13)
    ax_bottom.set_ylabel('Z-score', fontsize=13)
    ax_bottom.set_xlim(radii[0], radii[-1])
    ax_bottom.grid(True, alpha=0.3, linestyle='--')
    ax_bottom.legend(fontsize=11, loc='lower right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")

def plot_significance_analysis(radii, sigma_deviation, max_sigma, 
                               max_sigma_idx, filename):
    """Plot statistical significance analysis."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Sigma deviation profile
    ax1.plot(radii, sigma_deviation, 'o-', color='darkred', 
             linewidth=2, markersize=6)
    ax1.axhline(0, color='k', linestyle='--', linewidth=1)
    ax1.axhline(-1, color='orange', linestyle=':', linewidth=1, 
                alpha=0.7, label='1σ threshold')
    ax1.axhline(-2, color='red', linestyle=':', linewidth=1, 
                alpha=0.7, label='2σ threshold')
    ax1.axhline(-3, color='darkred', linestyle=':', linewidth=1, 
                alpha=0.7, label='3σ threshold')
    
    # Mark maximum deviation
    ax1.plot(radii[max_sigma_idx], sigma_deviation[max_sigma_idx],
             'r*', markersize=20, label=f'Max: {max_sigma:.2f}σ')
    
    ax1.set_xlabel('Radial distance [°]', fontsize=12)
    ax1.set_ylabel('Deviation [σ]', fontsize=12)
    ax1.set_title('Statistical Significance: Cold Spot Temperature Deviation',
                  fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')
    
    # Panel 2: Histogram of all control region profiles
    # This shows the distribution of temperatures at max deviation radius
    ax2.text(0.5, 0.5, 
             f'Statistical Significance Summary\n\n'
             f'Maximum deviation: {abs(max_sigma):.2f}σ\n'
             f'At radius: {radii[max_sigma_idx]:.1f}°\n\n'
             f'Interpretation:\n'
             f'  < 2σ: Consistent with random fluctuation\n'
             f'  2-3σ: Marginally significant\n'
             f'  > 3σ: Significant anomaly\n\n'
             f'The Cold Spot shows {"significant" if abs(max_sigma) > 3 else "marginal" if abs(max_sigma) > 2 else "weak"} deviation\n'
             f'from typical CMB fluctuations.',
             transform=ax2.transAxes,
             fontsize=13,
             verticalalignment='center',
             horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def save_statistical_report(radii, coldspot_mean, control_mean, control_std,
                           sigma_deviation, max_sigma, max_sigma_idx, filename):
    """Save detailed statistical report to text file."""
    
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COLD SPOT STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("Comparison with Random Control Regions\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Cold Spot location:     l={COLDSPOT_L}°, b={COLDSPOT_B}°\n")
        f.write(f"  Number of control regions: {N_CONTROL_REGIONS}\n")
        f.write(f"  Galactic plane mask:    |b| > {GALACTIC_MASK}°\n")
        f.write(f"  Maximum radius:         {MAX_RADIUS}°\n")
        f.write(f"  Radial bin size:        {RADIAL_STEP}°\n\n")
        
        f.write("Key Findings:\n")
        f.write(f"  Maximum deviation:      {abs(max_sigma):.2f}σ\n")
        f.write(f"  At radius:              {radii[max_sigma_idx]:.1f}°\n")
        f.write(f"  Cold Spot temperature:  {coldspot_mean[max_sigma_idx]:.1f} μK\n")
        f.write(f"  Control mean:           {control_mean[max_sigma_idx]:.1f} μK\n")
        f.write(f"  Control std dev:        {control_std[max_sigma_idx]:.1f} μK\n\n")
        
        f.write("Statistical Interpretation:\n")
        if abs(max_sigma) > 3:
            f.write("  → SIGNIFICANT ANOMALY (>3σ)\n")
            f.write("     The Cold Spot represents a statistically significant\n")
            f.write("     deviation from typical CMB fluctuations.\n")
        elif abs(max_sigma) > 2:
            f.write("  → MARGINALLY SIGNIFICANT (2-3σ)\n")
            f.write("     The Cold Spot shows marginal evidence of anomaly.\n")
        else:
            f.write("  → CONSISTENT WITH FLUCTUATIONS (<2σ)\n")
            f.write("     The Cold Spot is within expected random variation.\n")
        
        f.write("\n")
        f.write("RT Model Context:\n")
        f.write("  A significant deviation supports the hypothesis that localized\n")
        f.write("  RT network defects can create observable temperature patterns\n")
        f.write("  distinct from standard quantum fluctuations in the CMB.\n\n")
        
        f.write("Radial Profile Data:\n")
        f.write("  Radius [°]  Cold Spot [μK]  Control [μK]  Deviation [σ]\n")
        f.write("  " + "-" * 60 + "\n")
        for i in range(len(radii)):
            f.write(f"  {radii[i]:8.2f}  {coldspot_mean[i]:13.1f}  "
                   f"{control_mean[i]:13.1f}  {sigma_deviation[i]:13.2f}\n")
    
    print(f"  ✓ Saved: {filename}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Perform Cold Spot statistical significance analysis."""
    
    print("=" * 70)
    print("COLD SPOT STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Comparison with random control regions")
    print("=" * 70)
    
    import os
    
    print(f"\nConfiguration:")
    print(f"  Cold Spot:            l={COLDSPOT_L}°, b={COLDSPOT_B}°")
    print(f"  Control regions:      {N_CONTROL_REGIONS}")
    print(f"  Galactic mask:        |b| > {GALACTIC_MASK}°")
    print(f"  Analysis radius:      0° - {MAX_RADIUS}°")
    print(f"  Radial bins:          {RADIAL_STEP}°")
    
    # ========================================================================
    # Step 1: Load SMICA map
    # ========================================================================
    print("\n[1/5] Loading Planck SMICA map...")
    
    if os.path.exists(SMICA_PATH):
        smica_map = load_smica_map(SMICA_PATH)
    elif os.path.exists(SMICA_PATH_ALT):
        smica_map = load_smica_map(SMICA_PATH_ALT)
    else:
        print("ERROR: SMICA map not found!")
        return
    
    # ========================================================================
    # Step 2: Compute Cold Spot profile
    # ========================================================================
    print("\n[2/5] Computing Cold Spot radial profile...")
    
    cs_radii, cs_mean, cs_std = compute_radial_ring_profile(
        smica_map, NSIDE,
        COLDSPOT_L, COLDSPOT_B,
        MAX_RADIUS, RADIAL_STEP
    )
    
    print(f"  Cold Spot profile computed: {len(cs_radii)} radial bins")
    print(f"  Temperature range: [{np.nanmin(cs_mean):.1f}, {np.nanmax(cs_mean):.1f}] μK")
    
    # ========================================================================
    # Step 3: Generate control regions
    # ========================================================================
    print("\n[3/5] Generating control regions...")
    
    control_coords = generate_control_regions(N_CONTROL_REGIONS, GALACTIC_MASK)
    
    print(f"  Generated {len(control_coords)} control coordinates")
    print(f"  Longitude range: {control_coords.l.min():.1f}° - {control_coords.l.max():.1f}°")
    print(f"  Latitude range: {control_coords.b.min():.1f}° - {control_coords.b.max():.1f}°")
    
    # ========================================================================
    # Step 4: Compute control profiles
    # ========================================================================
    print("\n[4/5] Computing control region profiles...")
    
    ctrl_radii, ctrl_profiles, ctrl_mean, ctrl_std = compute_control_profiles(
        smica_map, NSIDE, control_coords,
        MAX_RADIUS, RADIAL_STEP
    )
    
    print(f"  Control profiles computed")
    print(f"  Mean temperature range: [{np.nanmin(ctrl_mean):.1f}, {np.nanmax(ctrl_mean):.1f}] μK")
    print(f"  Typical std dev: {np.nanmean(ctrl_std):.1f} μK")
    
    # Save profiles to CSV
    df = pd.DataFrame({
        'radius_deg': ctrl_radii,
        'coldspot_mean': cs_mean,
        'coldspot_std': cs_std,
        'control_mean': ctrl_mean,
        'control_std': ctrl_std
    })
    df.to_csv(OUTPUT_FILES['profiles_csv'], index=False)
    print(f"  ✓ Saved: {OUTPUT_FILES['profiles_csv']}")
    
    # ========================================================================
    # Step 5: Calculate significance and create plots
    # ========================================================================
    print("\n[5/5] Calculating statistical significance...")
    
    sigma_dev, max_sigma, max_sigma_idx = calculate_significance(
        cs_mean, ctrl_mean, ctrl_std
    )
    
    print(f"\n  Statistical Results:")
    print(f"    Maximum deviation:  {abs(max_sigma):.2f}σ")
    print(f"    At radius:          {cs_radii[max_sigma_idx]:.1f}°")
    print(f"    Cold Spot temp:     {cs_mean[max_sigma_idx]:.1f} μK")
    print(f"    Control mean:       {ctrl_mean[max_sigma_idx]:.1f} μK")
    
    # Create visualizations
    print("\n  Creating visualizations...")
    
    plot_comparison(
        cs_radii, cs_mean, cs_std,
        ctrl_mean, ctrl_std,
        OUTPUT_FILES['comparison']
    )
    
    plot_significance_analysis(
        cs_radii, sigma_dev, max_sigma, max_sigma_idx,
        OUTPUT_FILES['significance']
    )
    
    # Save report
    print("\n  Generating statistical report...")
    save_statistical_report(
        cs_radii, cs_mean, ctrl_mean, ctrl_std,
        sigma_dev, max_sigma, max_sigma_idx,
        OUTPUT_FILES['report']
    )

    # Create combined radial analysis (paper format)
    print("\n  Creating combined radial analysis (paper format)...")
    plot_radial_analysis_combined(
        cs_radii, cs_mean, ctrl_mean, ctrl_std, sigma_dev,
        OUTPUT_FILES['radial_analysis']
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nOutput files:")
    for key, filename in OUTPUT_FILES.items():
        print(f"  - {filename}")
    
    print("\nConclusion:")
    if abs(max_sigma) > 3:
        print("  The Cold Spot shows SIGNIFICANT deviation (>3σ) from typical")
        print("  CMB fluctuations, supporting its classification as an anomaly.")
    elif abs(max_sigma) > 2:
        print("  The Cold Spot shows MARGINAL deviation (2-3σ), suggesting")
        print("  possible anomalous behavior requiring further investigation.")
    else:
        print("  The Cold Spot deviation (<2σ) is consistent with random")
        print("  CMB fluctuations within the expected statistical range.")
    
    print()


if __name__ == "__main__":
    main()