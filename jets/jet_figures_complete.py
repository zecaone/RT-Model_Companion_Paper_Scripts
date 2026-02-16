#!/usr/bin/env python3
# ============================================================
# RT Model Companion Paper - Quasar Jet Figures (Section 5)
# ============================================================
# Purpose
# -------
# Generate all 5 figures for Section 5 "Case Study: Relativistic Jets 
# as Topological Channels in the RT Framework":
#
# 1. jet_morphology_schematic.png - AGN jet structure overview
# 2. rt_jet_channel_schematic.png - RT network with aligned links
# 3. rt_velocity_profile.png - Velocity profile RT vs MHD
# 4. rt_jet_termination.png - Jet termination and hot spots
# 5. m87_rt_overlay.png - M87 with RT channel overlay
#
# Physical Context
# ---------------
# Jets as topological channels in discrete RT-network structure.
# Frame-dragging near rotating black holes creates preferred pathways
# with enhanced stiffness and coherent link alignment.
#
# Dependencies
# -----------
# numpy, matplotlib
#
# Usage
# -----
#   python jet_figures_complete.py
#
# Output: figures/ directory with 5 PNG files (300 DPI)
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Wedge, Rectangle
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Create output directory
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

# Color scheme (scientific, colorblind-friendly)
COLOR_JET = '#1976D2'      # Blue for jet
COLOR_BH = '#212121'       # Black for black hole
COLOR_DISK = '#FF6F00'     # Orange for accretion disk
COLOR_CHANNEL = '#FBC02D'  # Yellow for RT channel
COLOR_RT_LINK = '#D32F2F'  # Red for RT links
COLOR_AMBIENT = '#757575'  # Gray for ambient
COLOR_SHOCK = '#E64A19'    # Red-orange for shocks

# ============================================================
# FIGURE 1: JET MORPHOLOGY SCHEMATIC
# ============================================================

def plot_jet_morphology_schematic():
    """
    Three-panel schematic showing AGN jet structure:
    - Left: Launching region (BH + accretion disk)
    - Center: Acceleration and collimation zone
    - Right: Extended jet with hot spots
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # --- PANEL 1: LAUNCHING REGION ---
    ax1 = axes[0]
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    
    # Black hole (event horizon)
    bh = Circle((0, 0), 0.3, color=COLOR_BH, zorder=10)
    ax1.add_patch(bh)
    
    # Accretion disk
    disk = Ellipse((0, 0), 2.0, 0.4, color=COLOR_DISK, alpha=0.6, zorder=5)
    ax1.add_patch(disk)
    
    # Jet base (cone emerging from poles)
    theta_open = 15  # opening angle in degrees
    for sign in [1, -1]:
        # Upper/lower jet cone
        cone = Wedge((0, 0), 2.5, 90 - theta_open/2 + sign*90, 
                     90 + theta_open/2 + sign*90,
                     color=COLOR_JET, alpha=0.5, zorder=3)
        ax1.add_patch(cone)
    
    # Annotations
    ax1.annotate('Black Hole', xy=(0, 0), xytext=(1.5, -1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, weight='bold')
    ax1.annotate('Accretion\nDisk', xy=(1.0, 0), xytext=(2.0, -2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, weight='bold')
    ax1.text(0, 2.8, 'Jet Launch', ha='center', fontsize=9, style='italic')
    ax1.text(0, -3.5, r'$r \sim 1-10\,r_g$', ha='center', fontsize=9)
    
    ax1.axis('off')
    ax1.set_title('(a) Launching Region', fontsize=12, weight='bold', pad=10)
    
    # --- PANEL 2: ACCELERATION ZONE ---
    ax2 = axes[1]
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 10)
    
    # Collimating jet (narrowing cone)
    y_base = 0.5
    y_top = 9.5
    width_base = 1.5
    width_top = 0.5
    
    # Draw jet as polygon
    jet_x = [-width_base, width_base, width_top, -width_top]
    jet_y = [y_base, y_base, y_top, y_top]
    jet_polygon = plt.Polygon(list(zip(jet_x, jet_y)), 
                              color=COLOR_JET, alpha=0.4, zorder=2)
    ax2.add_patch(jet_polygon)
    
    # Velocity arrows (increasing with height)
    for i, y in enumerate(np.linspace(2, 8, 4)):
        arrow_length = 0.3 + i * 0.15
        ax2.arrow(0, y, 0, arrow_length, head_width=0.3, head_length=0.2,
                 fc=COLOR_JET, ec=COLOR_JET, lw=2, zorder=5)
    
    # Magnetic field lines (helical suggestion)
    for x_sign in [-1, 1]:
        x_pos = x_sign * 0.8
        for y in np.linspace(1, 9, 8):
            ax2.plot([x_pos - 0.1, x_pos + 0.1], [y, y], 
                    color='gray', lw=0.8, alpha=0.5)
    
    # Annotations
    ax2.text(0, 0.2, 'Base', ha='center', fontsize=9, weight='bold')
    ax2.text(0, 9.8, 'Collimated', ha='center', fontsize=9, weight='bold')
    ax2.text(1.8, 5, r'$\Gamma$ increases', rotation=90, va='center',
            fontsize=9, style='italic')
    ax2.text(0, -1.2, r'$r \sim 10^2-10^4\,r_g$', ha='center', fontsize=9)
    
    ax2.axis('off')
    ax2.set_title('(b) Acceleration Zone', fontsize=12, weight='bold', pad=10)
    
    # --- PANEL 3: EXTENDED JET + HOT SPOTS ---
    ax3 = axes[2]
    ax3.set_aspect('equal')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(0, 12)
    
    # Long collimated jet
    jet_x = [-0.3, 0.3, 0.3, -0.3]
    jet_y = [0, 0, 10, 10]
    jet_polygon = plt.Polygon(list(zip(jet_x, jet_y)),
                             color=COLOR_JET, alpha=0.4, zorder=2)
    ax3.add_patch(jet_polygon)
    
    # Hot spot (termination shock)
    hotspot = Circle((0, 10.5), 0.6, color=COLOR_SHOCK, 
                     alpha=0.7, zorder=5)
    ax3.add_patch(hotspot)
    
    # Radio lobes (extended emission)
    for x_sign in [-1, 1]:
        lobe = Ellipse((x_sign * 2, 10.5), 2.5, 1.5, 
                      color=COLOR_DISK, alpha=0.3, zorder=1)
        ax3.add_patch(lobe)
    
    # Counter-jet (faint, for symmetry)
    counter_y = [-0.5, -3]
    ax3.plot([0, 0], counter_y, color=COLOR_JET, 
            lw=3, alpha=0.3, linestyle='--')
    
    # Annotations
    ax3.annotate('Hot Spot', xy=(0, 10.5), xytext=(1.5, 11.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, weight='bold')
    ax3.annotate('Radio Lobe', xy=(2, 10.5), xytext=(3.2, 9.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, weight='bold')
    ax3.text(0.5, 5, 'Jet', fontsize=10, weight='bold', rotation=90, va='center')
    ax3.text(0, -4, r'$r \gtrsim 10^4\,r_g$ (kpc-Mpc scales)', 
            ha='center', fontsize=9)
    
    ax3.axis('off')
    ax3.set_title('(c) Extended Jet', fontsize=12, weight='bold', pad=10)
    
    # Overall title
    fig.suptitle('AGN Jet Structure: From Launching to Termination', 
                fontsize=14, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    outfile = os.path.join(outdir, "jet_morphology_schematic.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ============================================================
# FIGURE 2: RT JET CHANNEL SCHEMATIC
# ============================================================

def plot_rt_jet_channel_schematic():
    """
    Schematic of RT-network structure near rotating black hole.
    Shows frame-dragging aligning RT links into topological channel.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # --- ROTATING BLACK HOLE ---
    bh = Circle((0, 0), 0.5, color=COLOR_BH, zorder=10)
    ax.add_patch(bh)
    
    # Spin indicator (curved arrow)
    from matplotlib.patches import FancyArrowPatch
    spin_arrow = FancyArrowPatch((0.3, 0.3), (-0.3, 0.3),
                                connectionstyle="arc3,rad=.5",
                                arrowstyle='->', mutation_scale=20,
                                lw=2, color='white', zorder=15)
    ax.add_patch(spin_arrow)
    ax.text(0, 0, r'$a$', color='white', fontsize=12, 
           weight='bold', ha='center', va='center', zorder=16)
    
    # --- ERGOSPHERE (optional, dashed ellipse) ---
    ergo = Ellipse((0, 0), 2.0, 1.5, fill=False, 
                   edgecolor='gray', linestyle=':', lw=1.5, alpha=0.6)
    ax.add_patch(ergo)
    ax.text(1.2, 0, 'Ergosphere', fontsize=8, style='italic', color='gray')
    
    # --- AMBIENT RT-NETWORK (random links) ---
    np.random.seed(42)
    n_ambient = 80
    for i in range(n_ambient):
        # Random positions outside channel
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(2.5, 5)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # Random link orientation
        link_angle = np.random.uniform(0, 2*np.pi)
        dx = 0.3 * np.cos(link_angle)
        dy = 0.3 * np.sin(link_angle)
        
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
               color=COLOR_AMBIENT, lw=0.8, alpha=0.4, zorder=1)
    
    # --- RT-CHANNEL (aligned links along axis) ---
    # Channel cone
    for y_sign in [1, -1]:
        channel_cone = Wedge((0, 0), 4.5, 90 - 10 + (1-y_sign)*90, 
                            90 + 10 + (1-y_sign)*90,
                            color=COLOR_CHANNEL, alpha=0.3, zorder=2)
        ax.add_patch(channel_cone)
    
    # Aligned RT links inside channel
    n_channel = 40
    for y_sign in [1, -1]:
        for i in range(n_channel // 2):
            r = np.random.uniform(1.0, 4.0)
            theta_offset = np.random.uniform(-8, 8) * np.pi/180  # small angle spread
            y = y_sign * r * np.cos(theta_offset)
            x = r * np.sin(theta_offset)
            
            # Links aligned vertically (along channel)
            dy = 0.25
            dx = 0.05 * np.random.randn()  # small horizontal wiggle
            
            ax.plot([x - dx, x + dx], [y - dy, y + dy],
                   color=COLOR_RT_LINK, lw=1.2, alpha=0.7, zorder=3)
    
    # --- FRAME-DRAGGING ARROWS ---
    for r in [1.5, 2.5, 3.5]:
        # Curved arrows showing rotational dragging
        arrow_azimuth = FancyArrowPatch((r*0.7, r*0.7), (-r*0.7, r*0.7),
                                       connectionstyle=f"arc3,rad=.3",
                                       arrowstyle='->', mutation_scale=15,
                                       lw=1.5, color='blue', alpha=0.6, zorder=4)
        ax.add_patch(arrow_azimuth)
    
    ax.text(0, 3.8, 'Frame-dragging', fontsize=10, 
           style='italic', color='blue', ha='center')
    
    # --- ANNOTATIONS ---
    ax.annotate('Rotating\nBlack Hole', xy=(0, -0.5), xytext=(-2.5, -2.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, weight='bold')
    
    ax.annotate('RT Channel\n(aligned links)', xy=(0.5, 3.0), xytext=(2.5, 4.0),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, weight='bold', color=COLOR_CHANNEL)
    
    ax.annotate('Ambient Network\n(random links)', xy=(3.5, 1.5), xytext=(4.5, 0.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, weight='bold', color=COLOR_AMBIENT)
    
    # Title
    ax.set_title('RT-Network Structure Near Rotating Black Hole\n' + 
                'Frame-dragging aligns links → Topological channel forms',
                fontsize=13, weight='bold', pad=15)
    
    ax.axis('off')
    plt.tight_layout()
    
    outfile = os.path.join(outdir, "rt_jet_channel_schematic.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ============================================================
# FIGURE 3: RT VELOCITY PROFILE
# ============================================================

def plot_rt_velocity_profile():
    """
    Radial velocity profile v(r) comparing RT model to MHD expectations.
    RT: steep initial acceleration → asymptotic approach to c
    MHD: gradual acceleration
    """
    # Radial coordinate (in gravitational radii)
    r_over_rg = np.logspace(1, 6, 500)  # 10 to 10^6 r_g
    
    # --- RT MODEL (Equation 5.16 from paper) ---
    r_acc = 300  # characteristic acceleration scale [r_g]
    # v/c = sqrt(1 - (r_acc/r)^2), but v=0 for r < r_acc (unphysical region)
    v_over_c_RT = np.zeros_like(r_over_rg)
    mask = r_over_rg >= r_acc
    v_over_c_RT[mask] = np.sqrt(1 - (r_acc / r_over_rg[mask])**2)
    
    # --- MHD MODEL (phenomenological: slower, gradual acceleration) ---
    # Simplified: v/c ~ tanh(r / r_mhd)
    r_mhd = 1000  # MHD acceleration scale (longer than RT)
    v_over_c_MHD = np.tanh(r_over_rg / r_mhd)
    
    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # RT curve
    ax.plot(r_over_rg, v_over_c_RT, linewidth=2.5, color=COLOR_JET,
           label='RT Model (topological channel)', zorder=3)
    
    # MHD curve
    ax.plot(r_over_rg, v_over_c_MHD, linewidth=2.5, color=COLOR_DISK,
           linestyle='--', label='MHD (magnetic acceleration)', zorder=2)
    
    # --- ANNOTATIONS ---
    # Mark acceleration scale
    ax.axvline(r_acc, color='gray', linestyle=':', alpha=0.6, zorder=1)
    ax.text(r_acc * 1.2, 0.3, r'$r_{\mathrm{acc}} \sim 300\,r_g$',
           fontsize=10, style='italic', color='gray')
    
    # Asymptotic line (v → c)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.4, zorder=1)
    ax.text(1e5, 1.02, r'$v \to c$', fontsize=10, style='italic')
    
    # Key regions
    ax.axvspan(10, 100, alpha=0.1, color='red', zorder=0)
    ax.text(30, 0.05, 'Launch', ha='center', fontsize=9, style='italic')
    
    ax.axvspan(100, 10000, alpha=0.1, color='orange', zorder=0)
    ax.text(1000, 0.05, 'Acceleration', ha='center', fontsize=9, style='italic')
    
    ax.axvspan(10000, 1e6, alpha=0.1, color='green', zorder=0)
    ax.text(1e5, 0.05, 'Collimated', ha='center', fontsize=9, style='italic')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel(r'Radial distance $r/r_g$', fontsize=13)
    ax.set_ylabel(r'Bulk velocity $v/c$', fontsize=13)
    ax.set_title('Jet Velocity Profile: RT vs MHD Models', 
                fontsize=14, weight='bold', pad=15)
    
    ax.set_xlim(10, 1e6)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    outfile = os.path.join(outdir, "rt_velocity_profile.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ============================================================
# FIGURE 4: RT JET TERMINATION
# ============================================================

def plot_rt_jet_termination():
    """
    Schematic of jet termination showing transition from coherent 
    RT channel to disrupted network at ISM interaction.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 12)
    ax.set_ylim(-3, 3)
    
    # --- LEFT: COHERENT RT CHANNEL ---
    # Jet cone (coherent channel)
    jet_left = Rectangle((0, -0.5), 5, 1.0, 
                         color=COLOR_JET, alpha=0.4, zorder=2)
    ax.add_patch(jet_left)
    
    # Aligned RT links
    np.random.seed(42)
    for x in np.linspace(0.5, 4.5, 30):
        for y in np.random.uniform(-0.4, 0.4, 3):
            dy = 0.15
            dx = 0.03 * np.random.randn()
            ax.plot([x - dx, x + dx], [y - dy, y + dy],
                   color=COLOR_RT_LINK, lw=1.2, alpha=0.8, zorder=3)
    
    ax.text(2.5, -1.5, 'Coherent RT Channel', fontsize=11, 
           weight='bold', ha='center', color=COLOR_JET)
    ax.text(2.5, -2.0, '(aligned links)', fontsize=9, 
           ha='center', style='italic', color=COLOR_JET)
    
    # --- CENTER: TERMINATION SHOCK ---
    # Shock front (vertical line with jagged edge)
    shock_x = 5.5
    shock_y = np.linspace(-1.5, 1.5, 20)
    shock_x_jag = shock_x + 0.1 * np.sin(shock_y * 5)
    ax.plot(shock_x_jag, shock_y, color=COLOR_SHOCK, 
           lw=3, alpha=0.8, zorder=5)
    
    # Hot spot (bright emission region)
    hotspot = Circle((shock_x, 0), 0.6, color=COLOR_SHOCK, 
                    alpha=0.6, zorder=4)
    ax.add_patch(hotspot)
    
    # Shock waves (expanding circles)
    for r in [0.8, 1.2, 1.6]:
        shock_wave = Circle((shock_x, 0), r, fill=False,
                           edgecolor=COLOR_SHOCK, linestyle='--',
                           lw=1.5, alpha=0.4, zorder=3)
        ax.add_patch(shock_wave)
    
    ax.text(shock_x, -2.2, 'Termination\nShock', fontsize=11,
           weight='bold', ha='center', color=COLOR_SHOCK)
    
    # --- RIGHT: DISRUPTED NETWORK (ISM) ---
    # Radio lobe (extended emission)
    lobe = Ellipse((9, 0), 4, 3, color=COLOR_DISK, alpha=0.3, zorder=1)
    ax.add_patch(lobe)
    
    # Disordered RT links
    for i in range(80):
        x = np.random.uniform(6.5, 11)
        y = np.random.uniform(-2.5, 2.5)
        
        link_angle = np.random.uniform(0, 2*np.pi)
        dx = 0.2 * np.cos(link_angle)
        dy = 0.2 * np.sin(link_angle)
        
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
               color=COLOR_AMBIENT, lw=0.8, alpha=0.5, zorder=2)
    
    # Synchrotron emission (wavy lines)
    for i in range(5):
        y_em = np.random.uniform(-1.5, 1.5)
        x_em = np.linspace(7, 10, 20)
        y_wave = y_em + 0.1 * np.sin((x_em - 7) * 3)
        ax.plot(x_em, y_wave, color=COLOR_DISK, 
               lw=1.0, alpha=0.6, linestyle=':', zorder=4)
    
    ax.text(9, -2.2, 'Radio Lobe', fontsize=11,
           weight='bold', ha='center', color=COLOR_DISK)
    ax.text(9, -2.7, '(synchrotron emission)', fontsize=9,
           ha='center', style='italic', color=COLOR_DISK)
    
    # --- AMBIENT ISM LABEL ---
    ax.text(9, 2.5, 'ISM/IGM\n(ambient medium)', fontsize=10,
           ha='center', style='italic', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.5))
    
    # Title
    ax.set_title('Jet Termination in RT Framework\n' +
                'Coherent channel disruption → Shock heating & synchrotron emission',
                fontsize=13, weight='bold', pad=15)
    
    ax.axis('off')
    plt.tight_layout()
    
    outfile = os.path.join(outdir, "rt_jet_termination.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ============================================================
# FIGURE 5: M87 RT OVERLAY
# ============================================================

def plot_m87_rt_overlay():
    """
    Simplified M87-inspired visualization with RT channel overlay.
    Shows photon ring, black hole shadow, and RT topological channel.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    
    # --- BACKGROUND: ACCRETION DISK (stylized) ---
    # Disk as gradient circles
    for r in np.linspace(0.5, 3.0, 15):
        alpha_val = 0.4 * (1 - (r - 0.5) / 2.5)
        disk_ring = Circle((0, 0), r, fill=False, 
                          edgecolor=COLOR_DISK, lw=2, alpha=alpha_val)
        ax.add_patch(disk_ring)
    
    # --- BLACK HOLE SHADOW ---
    bh_shadow = Circle((0, 0), 1.5, color='black', zorder=5)
    ax.add_patch(bh_shadow)
    
    # --- PHOTON RING (EHT-style) ---
    # Asymmetric brightness (Doppler beaming)
    theta = np.linspace(0, 2*np.pi, 100)
    r_ring = 2.5
    x_ring = r_ring * np.cos(theta)
    y_ring = r_ring * np.sin(theta)
    
    # Brightness varies with angle (brighter on approaching side)
    brightness = 0.3 + 0.5 * (1 + np.cos(theta - np.pi/4))
    brightness = np.clip(brightness, 0.0, 1.0)  # ensure valid alpha range
    
    for i in range(len(theta) - 1):
        ax.plot(x_ring[i:i+2], y_ring[i:i+2], 
               color='yellow', lw=3, alpha=brightness[i], zorder=6)
    
    # Photon Ring annotation - positioned to side with arrow, high contrast
    ax.annotate('Photon Ring at ~5$r_g$\n(EHT observation)', 
               xy=(2.2, 1.5), xytext=(4.2, 3.5),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
               fontsize=10, weight='bold', color='black',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='yellow', linewidth=2, alpha=0.95),
               ha='center', zorder=10)
    
    # --- RT JET CHANNEL OVERLAY ---
    # Channel cone (upper jet)
    for y_sign in [1, -1]:
        channel_cone = Wedge((0, 0), 5.5, 90 - 8 + (1-y_sign)*90,
                            90 + 8 + (1-y_sign)*90,
                            color=COLOR_CHANNEL, alpha=0.4, 
                            edgecolor=COLOR_CHANNEL, lw=2, zorder=7)
        ax.add_patch(channel_cone)
    
    # Channel boundary (emphasized)
    theta_cone = np.linspace(82, 98, 50) * np.pi/180
    for sign in [1, -1]:
        x_cone = 5.5 * np.cos(theta_cone)
        y_cone = sign * 5.5 * np.sin(theta_cone)
        ax.plot(x_cone, y_cone, color=COLOR_CHANNEL, 
               lw=2.5, alpha=0.9, zorder=8)
    
    # RT link structure inside channel (simplified)
    for y_sign in [1, -1]:
        for i in range(15):
            r = np.random.uniform(2.0, 5.0)
            theta_offset = np.random.uniform(-6, 6) * np.pi/180
            y = y_sign * r * np.cos(theta_offset)
            x = r * np.sin(theta_offset)
            
            dy = 0.3
            dx = 0.05 * np.random.randn()
            ax.plot([x - dx, x + dx], [y - dy, y + dy],
                   color=COLOR_RT_LINK, lw=1.5, alpha=0.6, zorder=7)
    
    # --- SCALE INDICATORS ---
    # Event Horizon (dashed circle at ~1 r_g) - HIGH Z-ORDER to be visible over shadow
    r_horizon = 0.5  # ~1 r_g in plot units
    horizon_circle = Circle((0, 0), r_horizon, fill=False,
                           edgecolor='white', linestyle='--',
                           lw=2, alpha=0.9, zorder=11)  # INCREASED z-order
    ax.add_patch(horizon_circle)
    
    # Gravitational radius markers for accretion disk
    for r_g_mult in [1, 3, 5]:
        r_circ = r_g_mult * 0.5  # 1 r_g ~ 0.5 units in plot
        scale_circle = Circle((0, 0), r_circ, fill=False,
                             edgecolor='white', linestyle=':', 
                             lw=1, alpha=0.4, zorder=4)
        ax.add_patch(scale_circle)
    
    # Scale labels with arrows pointing to circles
    # Event Horizon label - arrow points to nearest point on dashed circle (not through center)
    # Label is at (-4.0, -4.0), so nearest point on circle is at angle ~225° (bottom-left)
    angle_horizon = 225 * np.pi / 180  # bottom-left direction
    x_horizon_target = r_horizon * np.cos(angle_horizon)
    y_horizon_target = r_horizon * np.sin(angle_horizon)
    
    ax.annotate('Event Horizon\n(~1$r_g$)', 
               xy=(x_horizon_target, y_horizon_target), xytext=(-4.0, -4.0),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
               fontsize=9, weight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='#2a2a3e', 
                        edgecolor='white', linewidth=1.5, alpha=0.9),
               zorder=12)  # High z-order so arrow is above black shadow
    
    # 3 r_g marker with radial arrow
    r_3rg = 1.5
    angle_3rg = 25 * np.pi / 180  # angle for marker position
    x_3rg = r_3rg * np.cos(angle_3rg)
    y_3rg = r_3rg * np.sin(angle_3rg)
    x_label_3rg = 4.0
    y_label_3rg = 1.5
    
    # Arrow from label to circle
    ax.annotate('', xy=(x_3rg, y_3rg), xytext=(x_label_3rg - 0.3, y_label_3rg),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(x_label_3rg, y_label_3rg, '3$r_g$', fontsize=10, color='white', 
           weight='bold', ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='#2a2a3e', 
                    edgecolor='white', linewidth=1, alpha=0.8))
    
    # 5 r_g marker with radial arrow (pointing to yellow photon ring)
    r_5rg = 2.5
    angle_5rg = -25 * np.pi / 180
    x_5rg = r_5rg * np.cos(angle_5rg)
    y_5rg = r_5rg * np.sin(angle_5rg)
    x_label_5rg = 4.0
    y_label_5rg = -1.5
    
    # Arrow from label to yellow circle
    ax.annotate('', xy=(x_5rg, y_5rg), xytext=(x_label_5rg - 0.3, y_label_5rg),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(x_label_5rg, y_label_5rg, '5$r_g$', fontsize=10, color='white',
           weight='bold', ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='#2a2a3e',
                    edgecolor='white', linewidth=1, alpha=0.8))
    
    # --- ANNOTATIONS ---
    # RT Channel annotation with high-contrast box
    ax.annotate('RT Topological\nChannel', xy=(0.5, 4.0), xytext=(2.5, 5.2),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=3, 
                             mutation_scale=20),
               fontsize=11, weight='bold', color='black',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='yellow', linewidth=2, alpha=0.95))
    
    # Black hole shadow annotation - radial from outside
    # Position on right side, pointing radially inward to shadow edge
    shadow_radius = 1.5
    angle_shadow = -70 * np.pi / 180  # angle for radial arrow (lower right)
    x_shadow_edge = shadow_radius * np.cos(angle_shadow)
    y_shadow_edge = shadow_radius * np.sin(angle_shadow)
    x_shadow_label = 3.5
    y_shadow_label = -3.0
    
    ax.annotate('Black Hole\nShadow', 
               xy=(x_shadow_edge, y_shadow_edge), xytext=(x_shadow_label, y_shadow_label),
               arrowprops=dict(arrowstyle='->', color='cyan', lw=2,
                             mutation_scale=15),
               fontsize=10, weight='bold', color='cyan')
    
    # Info box with better contrast
    info_text = 'M87*\n' + \
                r'$M \sim 6.5 \times 10^9\,M_\odot$' + '\n' + \
                r'$a \sim 0.9$ (high spin)'
    ax.text(-5, 5, info_text, fontsize=9, color='black',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='orange', linewidth=2, alpha=0.95),
           verticalalignment='top')
    
    # Background (dark blue-gray, space-like but not pure black)
    bg_color = '#1a1a2e'  # Dark blue-gray for better contrast
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    # Title with better visibility
    ax.text(0, 6.3, 'M87 Jet: EHT Observation + RT Channel Interpretation',
           fontsize=13, weight='bold', ha='center', color='white',
           bbox=dict(boxstyle='round', facecolor='#2a2a3e', 
                    edgecolor='white', linewidth=1, alpha=0.8, pad=0.5))
    
    ax.axis('off')
    plt.tight_layout()
    
    outfile = os.path.join(outdir, "m87_rt_overlay.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Generate all 5 jet figures for companion paper Section 5"""
    
    print("\n" + "="*60)
    print("RT Companion Paper: Quasar Jet Figures Generator")
    print("Section 5: Jets as Topological Channels")
    print("="*60 + "\n")
    
    print("Generating Figure 1: Jet Morphology Schematic...")
    plot_jet_morphology_schematic()
    
    print("\nGenerating Figure 2: RT Jet Channel Schematic...")
    plot_rt_jet_channel_schematic()
    
    print("\nGenerating Figure 3: RT Velocity Profile...")
    plot_rt_velocity_profile()
    
    print("\nGenerating Figure 4: RT Jet Termination...")
    plot_rt_jet_termination()
    
    print("\nGenerating Figure 5: M87 RT Overlay...")
    plot_m87_rt_overlay()
    
    print("\n" + "="*60)
    print("✓ All 5 jet figures generated successfully!")
    print(f"✓ Output directory: {os.path.abspath(outdir)}/")
    print("="*60 + "\n")
    
    print("Summary:")
    print("-" * 60)
    print("  jet_morphology_schematic.png    - AGN jet structure (3 panels)")
    print("  rt_jet_channel_schematic.png    - RT network with frame-dragging")
    print("  rt_jet_velocity_profile.png     - v(r) RT vs MHD comparison")
    print("  rt_jet_termination.png          - Shock & hot spot formation")
    print("  m87_rt_overlay.png              - M87 with RT channel overlay")
    print("-" * 60)


if __name__ == "__main__":
    main()
