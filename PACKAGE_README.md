# COSMOS: Simplified Odor Simulation Package

[![PyPI version](https://badge.fury.io/py/cosmos-odor.svg)](https://badge.fury.io/py/cosmos-odor)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

COSMOS (Configurable Odor Simulation Model over Scalable Spaces) is a Python package for realistic odor time series simulation. This package provides a dramatically simplified interface for real-time odor concentration prediction across different environments.

## 🚀 Ultra-Simple Interface

**Before (16+ lines of complex setup):**
```python
import sys
sys.path.append('COSMOS')
from cosmos_tracking import CosmosFast
import numpy as np
import pandas as pd

dirname = 'COSMOS/data/hws/'
hmap_data = np.load(str(dirname) + "hmap.npz")
fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')

predictor = CosmosFast(
    fitted_p_heatmap=hmap_data['fitted_heatmap'],
    xedges=hmap_data['xedges'],
    yedges=hmap_data['yedges'],
    fdf=fdf,
    fdf_nowhiff=fdf_nowhiff
)
```

**After (2 lines total):**
```python
import cosmos
model = cosmos.predictor('desert-hws')
```

## 📦 Installation & Setup

### Step 1: Install Package
```bash
git clone https://github.com/arunavanag591/COSMOS.git
cd COSMOS
pip install -e .
```

### Step 2: Download Data 
**Required:** Download data from [Dryad](http://datadryad.org/share/6ahtoddnVD7c3Tj2zKHLjVn3GTtAj-W6zqIYu9udpL4)

1. Extract the downloaded data 
2. Place the `data` folder in the COSMOS directory
3. For Rigolli model: Download additional files from [Zenodo](https://zenodo.org/records/15469831)
4. Place Rigolli files (`coordinates.mat`, `crosswind_v.mat`, etc.) in `data/rigolli/`

**Your directory should look like:**
```
COSMOS/
├── cosmos/          # Package code  
├── data/           # Downloaded from Dryad
│   ├── hws/
│   ├── lws/
│   ├── forest/
│   └── rigolli/    # Additional files from Zenodo
└── README.md
```

### Step 3: Use the Simplified Interface

```python
import cosmos

# Create predictor - handles all data loading automatically
model = cosmos.predictor('desert-hws')

# Simulate odor at position (x=1.0, y=0.5)
concentration = model.step_update(1.0, 0.5)
print(f"Odor concentration: {concentration}")
```

That's it! No manual data downloads, no path management, no complex imports.

## Real-time Simulation Example

```python
import cosmos
import numpy as np

# Initialize model
model = cosmos.predictor('desert-hws')

# Create a trajectory  
dt = 0.005
time = np.arange(0, 10, dt)
x_pos = np.sin(time * 2 * np.pi * 0.5) + 2
y_pos = np.sin(time * 2 * np.pi * 4)

# Simulate odor experience
odors = []
for x, y in zip(x_pos, y_pos):
    concentration = model.step_update(x, y, dt)
    odors.append(concentration)

print(f"Simulated {len(odors)} time steps")
```

## Available Models

- `'desert-hws'`: Desert environment with high wind speeds (3.5-6 m/s)
- `'desert-lws'`: Desert environment with low wind speeds  
- `'forest'`: Forest environment measurements
- `'rigolli'`: CFD-based simulation data

```python
# List all available models
models = cosmos.list_available_models()
for model in models:
    print(f"{model['name']}: {model['description']}")
```

## API Reference

### `cosmos.predictor(model_name)`

Create a COSMOS predictor for real-time odor simulation.

**Parameters:**
- `model_name` (str): Name of pre-trained model ('desert-hws', 'desert-lws', 'forest', 'rigolli')

**Returns:**
- CosmosFast object with `step_update(x, y, dt)` method

### `predictor.step_update(x, y, dt=0.005)`

Update odor concentration for current position.

**Parameters:**
- `x` (float): Current x position
- `y` (float): Current y position  
- `dt` (float): Time step in seconds (default: 0.005)

**Returns:**
- `float`: Odor concentration at current position

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0
- numba ≥ 0.56.0
- h5py ≥ 3.0.0

## Installation from Source

```bash
git clone https://github.com/arunavanag591/COSMOS.git
cd COSMOS
pip install -e .
```

## Citation

If you use COSMOS in your research, please cite:

```bibtex
@article{nag2025cosmos,
  title={COSMOS: A Data-Driven Probabilistic Time Series simulator for Chemical Plumes across Spatial Scales},
  author={Nag, Arunava and van Breugel, Floris},
  journal={arXiv preprint arXiv:2505.22436},
  year={2025}
}

```

## License

This project is dedicated to the public domain under CC0 1.0 Universal.

## Repository

Full source code and documentation: https://github.com/arunavanag591/COSMOS