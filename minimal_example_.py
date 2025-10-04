#!/usr/bin/env python3
"""
Prerequisites:
1. Download data from: http://datadryad.org/share/6ahtoddnVD7c3Tj2zKHLjVn3GTtAj-W6zqIYu9udpL4
2. Place 'data' folder in COSMOS directory
"""

import cosmos

try:
    # Single line to create predictor - replaces all the data loading complexity
    model = cosmos.predictor('desert-hws')

    # Single line to get odor concentration at any position
    concentration = model.step_update(1.0, 0.5)

    print(f"Odor concentration at position (1.0, 0.5): {concentration}")
    print("✓ COSMOS package working perfectly!")

except FileNotFoundError as e:
    print("❌ Data not found!")
    print(str(e))
    print("\nPlease download the required data files as described above.")

# That's it! No imports of numpy, pandas, h5py, matplotlib
# No manual data path handling, no loading hmap files, no DataFrame operations
# Just: import cosmos, create model, get concentration