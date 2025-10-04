#!/usr/bin/env python3
"""
Ultra-minimal COSMOS example showing the simplified interface.
This example replaces 20+ lines of complex setup with just 3 lines.
"""

import cosmos

# Single line to create predictor - replaces all the data loading complexity
model = cosmos.predictor('desert-hws')

# Single line to get odor concentration at any position
concentration = model.step_update(1.0, 0.5)

print(f"Odor concentration at position (1.0, 0.5): {concentration}")

# That's it! No imports of numpy, pandas, h5py, matplotlib
# No manual data path handling, no loading hmap files, no DataFrame operations
# Just: import cosmos, create model, get concentration