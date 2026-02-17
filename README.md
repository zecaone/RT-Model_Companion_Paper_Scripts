# RT Model Companion Paper - Code Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18674355.svg)](https://doi.org/10.5281/zenodo.18674355)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code accompanying the paper:

**"Observable Cosmological Effects and Testable Predictions in the Raumteilchen Model"**  
by Urs Krafzig (2026)

## Overview

This repository contains all Python scripts used to generate the 27 figures in the companion paper to *"The Raumteilchen Model: Foundations of a Discrete Spacetime Framework"*. The RT model provides a discrete, dynamically evolving framework for spacetime and matter, offering new insights into cosmological anomalies and testable predictions.

## Contents

- **11 Python scripts** generating **27 publication-quality figures**
- Analysis code for CMB anomalies (Cold Spot, Axis of Evil)
- GRB photon delay analysis with energy-dependent propagation
- Hubble tension consistency simulations
- Quasar jet morphology and topological channel models

## Repository Structure

```
rt-model-companion/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── FIGURE_MAPPING.txt          # Complete figure-to-script mapping
│
├── scripts/
│   ├── cmb/                    # CMB anomaly analysis (7 scripts, 13 figures)
│   │   ├── axis_of_evil_analysis.py
│   │   ├── axis_of_evil_planck_analysis.py
│   │   ├── aoe_sim_paper.py
│   │   ├── cold_spot_base_maps.py
│   │   ├── coldspot_radial_analysis.py
│   │   ├── generate_cold_spot_variants.py
│   │   └── simulate_multiple_rt_defects.py
│   │
│   ├── grb/                    # GRB analysis (2 scripts, 5 figures)
│   │   ├── grb_221009a_lightcurve.py
│   │   └── grb_analysis_complete.py
│   │
│   ├── hubble/                 # Hubble tension (1 script, 4 figures)
│   │   └── hubble_companion_figures.py
│   │
│   └── jets/                   # Quasar jets (1 script, 5 figures)
│       └── jet_figures_complete.py
│
└── figures/                    # Generated figures (created by scripts)
    ├── axis_of_evil_*.png
    ├── cold_spot_*.png
    ├── grb_*.png
    ├── hubble_*.png
    └── jet_*.png
```

## Installation

### Requirements

- Python 3.9+
- NumPy
- Matplotlib
- HEALPix (healpy) for CMB analyses
- SciPy
- Astropy (for cosmological calculations)

### Setup

```bash
# Clone repository
git clone https://github.com/zecaone/RT-Model_Companion_Paper_Scripts.git
cd RT-Model_Companion_Paper_Scripts

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate All Figures

```bash
# CMB Anomalies
python scripts/cmb/axis_of_evil_analysis.py
python scripts/cmb/axis_of_evil_planck_analysis.py
python scripts/cmb/aoe_sim_paper.py
python scripts/cmb/cold_spot_base_maps.py
python scripts/cmb/coldspot_radial_analysis.py
python scripts/cmb/generate_cold_spot_variants.py
python scripts/cmb/simulate_multiple_rt_defects.py

# GRB Analysis
python scripts/grb/grb_221009a_lightcurve.py
python scripts/grb/grb_analysis_complete.py

# Hubble Consistency
python scripts/hubble/hubble_companion_figures.py

# Quasar Jets
python scripts/jets/jet_figures_complete.py
```

All figures will be saved to the `figures/` directory in PNG format (300 DPI).

## Individual Script Details

### CMB Anomalies (Section 4.1)

#### Axis of Evil Analysis

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `axis_of_evil_analysis.py` | axis_of_evil_axes_comparison.png | Quadrupole/octupole alignment analysis |
| `axis_of_evil_planck_analysis.py` | axis_of_evil_planck_comparison.png | Planck SMICA data analysis |
| `aoe_sim_paper.py` | axis_of_evil_simulated_map.png | Simulated anisotropic RT network |

#### Cold Spot Analysis

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `cold_spot_base_maps.py` | cold_spot_original_map.png<br>cold_spot_damped_map.png<br>cold_spot_difference_map.png<br>cold_spot_histogram.png | Base CMB maps with RT defect damping |
| `coldspot_radial_analysis.py` | coldspot_radial_analysis.png | Radial temperature profiles + Z-scores |
| `generate_cold_spot_variants.py` | cold_spot_variant_A.png<br>cold_spot_variant_B.png<br>cold_spot_variant_C.png<br>cold_spot_variant_D.png | Spherical harmonic defect modeling (4 variants) |

#### Multi-Defect Simulation

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `simulate_multiple_rt_defects.py` | multi_defect_map.png | Three-defect RT network simulation |

**CMB Total: 7 scripts → 13 figures**

---

### GRB Analysis (Section 4.2)

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `grb_221009a_lightcurve.py` | grb_221009a_lightcurve.png | Multi-wavelength light curve (BOAT) |
| `grb_analysis_complete.py` | grb_fit_n1.png<br>grb_fit_n2.png<br>grb_redshift_scaling.png<br>grb_residuals.png | Energy-dependent delay analysis (n=1 vs n=2) |

**GRB Total: 2 scripts → 5 figures**

---

### Hubble Consistency (Section 4.3)

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `hubble_companion_figures.py` | hubble_tension_overview.png<br>hubble_sigma1e6.png<br>hubble_sigma1e4.png<br>hubble_sigma1e3.png | RT-network propagation effects on H₀ |

**Hubble Total: 1 script → 4 figures**

---

### Quasar Jets (Section 5)

| Script | Figures Generated | Description |
|--------|-------------------|-------------|
| `jet_figures_complete.py` | jet_morphology_schematic.png<br>rt_jet_channel_schematic.png<br>rt_jet_velocity_profile.png<br>rt_jet_termination.png<br>m87_rt_overlay.png | Topological channel interpretation |

**Jets Total: 1 script → 5 figures**

---

## Complete Figure List

### All 27 Figures

| # | Figure | Script | Section |
|---|--------|--------|---------|
| 1 | axis_of_evil_axes_comparison.png | axis_of_evil_analysis.py | 4.1.2 |
| 2 | axis_of_evil_planck_comparison.png | axis_of_evil_planck_analysis.py | 4.1.2 |
| 3 | axis_of_evil_simulated_map.png | aoe_sim_paper.py | 4.1.2 |
| 4 | cold_spot_original_map.png | cold_spot_base_maps.py | 4.1.1 |
| 5 | cold_spot_damped_map.png | cold_spot_base_maps.py | 4.1.1 |
| 6 | cold_spot_difference_map.png | cold_spot_base_maps.py | 4.1.1 |
| 7 | cold_spot_histogram.png | cold_spot_base_maps.py | 4.1.1 |
| 8 | coldspot_radial_analysis.png | coldspot_radial_analysis.py | 4.1.1 |
| 9 | cold_spot_variant_A.png | generate_cold_spot_variants.py | 4.1.1 |
| 10 | cold_spot_variant_B.png | generate_cold_spot_variants.py | 4.1.1 |
| 11 | cold_spot_variant_C.png | generate_cold_spot_variants.py | 4.1.1 |
| 12 | cold_spot_variant_D.png | generate_cold_spot_variants.py | 4.1.1 |
| 13 | multi_defect_map.png | simulate_multiple_rt_defects.py | 4.1.2 |
| 14 | grb_221009a_lightcurve.png | grb_221009a_lightcurve.py | 4.2.4 |
| 15 | grb_fit_n1.png | grb_analysis_complete.py | 4.2.3 |
| 16 | grb_fit_n2.png | grb_analysis_complete.py | 4.2.3 |
| 17 | grb_redshift_scaling.png | grb_analysis_complete.py | 4.2.3 |
| 18 | grb_residuals.png | grb_analysis_complete.py | 4.2.5 |
| 19 | hubble_tension_overview.png | hubble_companion_figures.py | 4.3.1 |
| 20 | hubble_sigma1e6.png | hubble_companion_figures.py | 4.3.3 |
| 21 | hubble_sigma1e4.png | hubble_companion_figures.py | 4.3.3 |
| 22 | hubble_sigma1e3.png | hubble_companion_figures.py | 4.3.3 |
| 23 | jet_morphology_schematic.png | jet_figures_complete.py | 5.1 |
| 24 | rt_jet_channel_schematic.png | jet_figures_complete.py | 5.3 |
| 25 | rt_jet_velocity_profile.png | jet_figures_complete.py | 5.4 |
| 26 | rt_jet_termination.png | jet_figures_complete.py | 5.5 |
| 27 | m87_rt_overlay.png | jet_figures_complete.py | 5.7 |

**Total: 11 scripts → 27 figures**

---

## Citation

If you use this code in your research, please cite:

### Paper
```bibtex
@article{krafzig2026rtcompanion,
  author = {Krafzig, Urs},
  title = {Observable Cosmological Effects and Testable Predictions in the Raumteilchen Model},
  journal = {preprint (submitted)},
  year = {2026},
  note = {Companion paper to the RT Model foundational framework},
  url = {https://github.com/zecaone/RT-Model_Companion_Paper_Scripts}
}
```

### Code
```bibtex
@software{krafzig2026rtcode,
  author = {Krafzig, Urs},
  title = {RT Model Companion Paper - Code Repository},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0},
  doi = {10.5281/zenodo.18674355},
  url = {https://github.com/zecaone/RT-Model_Companion_Paper_Scripts}
}
```

*Note: The paper DOI will be added once the manuscript is accepted for publication. The Zenodo DOI for the code will be generated automatically when the first release is created.*

---

## License

This code is released under the [MIT License](LICENSE).

## Reproducibility

All figures in the paper can be reproduced by running the scripts in this repository. The code has been tested with Python 3.9+ on Linux and macOS.

## Contact

For questions or issues, please:
- Open an [issue](https://github.com/zecaone/RT-Model_Companion_Paper_Scripts/issues)
- Contact: urs.krafzig@gmail.com

## Acknowledgments

This research was conducted as an independent side project alongside full-time employment. Special thanks to the communities developing HEALPix, Matplotlib, and scientific Python tools.

## Related Publications

- **Foundational Paper**: Krafzig, U. (2026). "The Raumteilchen Model: Foundations of a Discrete Spacetime Framework" (in preparation)
- **Companion Paper**: Krafzig, U. (2026). "Observable Cosmological Effects and Testable Predictions in the Raumteilchen Model" (in preparation)

## Version History

- **v1.0** (2026-02-17): Initial release with all 11 scripts generating 27 figures

---

**Last updated:** February 2026
