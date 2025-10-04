# COSMOS Package Deployment Guide

## Summary of Changes

We have successfully transformed the COSMOS codebase into a professional Python package that addresses all reviewer concerns:

### Before vs After

**Original Interface (16+ lines):**
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
    fitted_p_heatmap=hmap_data['fitted_heatmap'],  # Inconsistent naming!
    xedges=hmap_data['xedges'],
    yedges=hmap_data['yedges'],
    fdf=fdf,
    fdf_nowhiff=fdf_nowhiff
)
```

**New Interface (2 lines):**
```python
import cosmos
model = cosmos.predictor('desert-hws')
```

**Usage (1 line):**
```python
concentration = model.step_update(x, y)
```

## Key Improvements

### 1. ✅ Package Structure
- Created proper Python package with `cosmos/` directory
- Added `__init__.py`, `core.py`, `data_loader.py`
- Proper `setup.py` and `pyproject.toml` for PyPI distribution

### 2. ✅ Simplified API
- Single function interface: `cosmos.predictor(model_name)`
- Handles all data loading internally
- Consistent naming across all models
- No manual path management required

### 3. ✅ Data Management  
- Unified data interface for all 4 models
- Automatic fallback for naming inconsistencies
- Built-in data validation
- Models included in package distribution

### 4. ✅ User Experience
- Minimal imports (just `import cosmos`)
- Clear error messages
- Available models listing
- Comprehensive documentation

### 5. ✅ Distribution Ready
- PyPI-ready package structure
- All dependencies properly specified
- Data files included in distribution
- Wheel and source distributions built

## Installation Methods

### Option 1: From PyPI (Recommended for end users)
```bash
pip install cosmos-odor
```

### Option 2: From source (Development)
```bash
git clone https://github.com/arunavanag591/COSMOS.git
cd COSMOS
pip install -e .
```

### Option 3: From wheel (Local testing)
```bash
pip install dist/cosmos_odor-1.0.0-py3-none-any.whl
```

## Usage Examples

### Ultra-minimal (3 lines total)
```python
import cosmos
model = cosmos.predictor('desert-hws')
concentration = model.step_update(1.0, 0.5)
```

### Real-time simulation
```python
import cosmos
import numpy as np

model = cosmos.predictor('desert-hws')

# Define trajectory
dt = 0.005
time = np.arange(0, 10, dt)
x_pos = np.sin(time * 0.5) + 2
y_pos = np.sin(time * 4)

# Simulate odor experience
odors = []
for x, y in zip(x_pos, y_pos):
    concentration = model.step_update(x, y, dt)
    odors.append(concentration)
```

### List available models
```python
import cosmos
models = cosmos.list_available_models()
for model in models:
    print(f"{model['name']}: {model['description']}")
```

## Available Models

1. **`'desert-hws'`** - Desert environment with high wind speeds (3.5-6 m/s)
2. **`'desert-lws'`** - Desert environment with low wind speeds  
3. **`'forest'`** - Forest environment measurements
4. **`'rigolli'`** - CFD-based Rigolli simulation data

## PyPI Deployment

### 1. Test the package locally
```bash
python3 -m build
pip install dist/cosmos_odor-1.0.0-py3-none-any.whl --force-reinstall
```

### 2. Upload to PyPI Test (optional)
```bash
pip install twine
twine upload --repository testpypi dist/*
```

### 3. Upload to PyPI Production
```bash
twine upload dist/*
```

### 4. Verify installation
```bash
pip install cosmos-odor
python3 -c "import cosmos; model = cosmos.predictor('desert-hws'); print('Works!')"
```

## Package Contents

```
cosmos/
├── __init__.py           # Main package interface
├── core.py              # CosmosFast class and predictor() function  
├── data_loader.py       # Unified data loading with fallbacks
└── data/                # Pre-trained models (included in package)
    ├── hws/            # High wind speed desert data
    ├── lws/            # Low wind speed desert data  
    ├── forest/         # Forest environment data
    └── rigolli/        # CFD simulation data

Examples:
├── ultra_minimal_example.py    # 3-line demo
├── example_minimal.py          # Complete trajectory example
└── cosmos_minimal_example.ipynb # Jupyter notebook demo

Package files:
├── setup.py            # Package setup script
├── pyproject.toml      # Modern package configuration
├── MANIFEST.in         # Files to include in distribution
├── PACKAGE_README.md   # Package-specific documentation
└── requirements.txt    # Dependencies
```

## Dependencies

**Core dependencies (automatically installed):**
- numpy >= 1.20.0
- pandas >= 1.3.0  
- scipy >= 1.7.0
- numba >= 0.56.0
- h5py >= 3.0.0

**Optional dependencies:**
- matplotlib >= 3.5.0 (for visualization examples)

## Addressing Reviewer Concerns

### ✅ "Code doesn't promote package usage"
**Before:** Users had to download entire repo, manually handle paths
**After:** Single `pip install cosmos-odor` + `import cosmos`

### ✅ "Encourages people to copy code" 
**Before:** Manual file copying and path management required
**After:** Clean package interface, no file management needed

### ✅ "Increases barrier to use"
**Before:** 16+ lines of setup code with manual data paths  
**After:** 2 lines total (`import cosmos` + `predictor()`)

### ✅ "Lose control over code"
**Before:** Users copy raw functions
**After:** Users import package, get updates via pip

### ✅ "Hard to promote reporting issues/PRs"
**Before:** No clear package identity
**After:** Clear package name, PyPI presence, GitHub integration

### ✅ "Data handling inconsistencies"
**Before:** Different key names across models (`fitted_heatmap` vs `fitted_p_heatmap`)
**After:** Automatic fallback handles all naming inconsistencies

### ✅ "Manual data download required"
**Before:** Users must manually download and place data files
**After:** All data included in package distribution

## Testing

The package has been tested with:
- ✅ Fresh installation from wheel
- ✅ Import and basic functionality  
- ✅ All 4 model types
- ✅ Real-time trajectory simulation
- ✅ Data integrity and naming fallbacks

## Next Steps

1. **Publish to PyPI:** Upload the built package to PyPI for public access
2. **Update paper:** Include package name in abstract and references  
3. **Documentation:** Link to package in README and documentation
4. **Examples:** Update existing notebooks to use new interface
5. **Citation:** Include package installation in citation examples

## Benefits Achieved

1. **16 lines → 2 lines:** Massive reduction in setup complexity
2. **Zero manual data management:** All handled automatically  
3. **Consistent interface:** Same API for all models
4. **Professional distribution:** PyPI-ready package
5. **Improved discoverability:** Clear package identity
6. **Future-proof:** Easy to update and maintain
7. **Better adoption:** Lower barrier to entry
8. **Credit tracking:** Package usage tracked via pip/PyPI

The COSMOS package now meets all professional Python package standards and addresses every concern raised by the reviewers.