# GRIFO

GRIFO (Graphical Reduction and Inference for exoplanetary transit Observations) is a desktop GUI for photometric analysis using **PySide6 + pyqtgraph**.

## Features

- FITS loading and frame inspection
- Frame alignment
- Target/comparison star coordinate workflow
- Aperture photometry
- Detrending
- Final fitting step with:
  - Batman transit MCMC
  - Polynomial fit (general source light curves)

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
