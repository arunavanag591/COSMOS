"""
Data loading utilities for COSMOS models.
Handles all the different dataset formats and naming inconsistencies.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Model configuration mapping
MODEL_CONFIGS = {
    'desert-hws': {
        'directory': 'hws',
        'hmap_file': 'hmap.npz',
        'hmap_key': 'fitted_heatmap',  # Note: inconsistent naming
        'whiff_file': 'whiff.h5',
        'nowhiff_file': 'nowhiff.h5',
        'description': 'Desert environment with high wind speeds (3.5-6 m/s)'
    },
    'desert-lws': {
        'directory': 'lws', 
        'hmap_file': 'lws_hmap_with_edges.npz',
        'hmap_key': 'fitted_p_heatmap',
        'whiff_file': 'whiff.h5',
        'nowhiff_file': 'nowhiff.h5',
        'description': 'Desert environment with low wind speeds'
    },
    'forest': {
        'directory': 'forest',
        'hmap_file': 'forest_hmap_with_edges.npz', 
        'hmap_key': 'fitted_p_heatmap',
        'whiff_file': 'forest.h5',
        'nowhiff_file': 'forest.h5',
        'description': 'Forest environment measurements'
    },
    'rigolli': {
        'directory': 'rigolli',
        'hmap_file': 'hmap.npz',
        'hmap_key': 'fitted_heatmap',  # Note: inconsistent naming  
        'whiff_file': 'whiff.h5',
        'nowhiff_file': 'nowhiff.h5',
        'description': 'CFD-based Rigolli simulation data'
    }
}

def get_data_directory():
    """Get the data directory path, trying multiple locations."""
    current_dir = Path(__file__).parent
    
    # Try package location first (when installed)
    package_data_dir = current_dir / "data"
    if package_data_dir.exists():
        return package_data_dir
    
    # Try development location (when running from repo)
    dev_data_dir = current_dir.parent / "data"
    if dev_data_dir.exists():
        return dev_data_dir
    
    # Try user's home directory (downloaded data)
    home_data_dir = Path.home() / ".cosmos" / "data"
    if home_data_dir.exists():
        return home_data_dir
    
    raise FileNotFoundError(
        "COSMOS data not found. Please download the data from Dryad and place it in the COSMOS directory:\n\n"
        "1. Download data from: http://datadryad.org/share/6ahtoddnVD7c3Tj2zKHLjVn3GTtAj-W6zqIYu9udpL4\n"
        "2. Extract and place the 'data' folder in the COSMOS home directory\n"
        "3. For Rigolli data, also download from: https://zenodo.org/records/15469831\n"
        "   and place coordinates.mat, crosswind_v.mat, downwind_v.mat, ground_data.mat,\n"
        "   nose_data.mat, vertical_v.mat in the data/rigolli/ location\n\n"
        "After setup, your directory should look like:\n"
        "COSMOS/\n"
        "├── cosmos/          # Package code\n"
        "└── data/           # Downloaded data\n"
        "    ├── hws/\n"
        "    ├── lws/\n" 
        "    ├── forest/\n"
        "    └── rigolli/\n"
    )

def list_available_models():
    """List all available pre-trained models."""
    try:
        data_dir = get_data_directory()
        available = []
        
        for model_name, config in MODEL_CONFIGS.items():
            model_dir = data_dir / config['directory']
            if model_dir.exists():
                required_files = [config['hmap_file'], config['whiff_file'], config['nowhiff_file']]
                if all((model_dir / f).exists() for f in required_files):
                    available.append({
                        'name': model_name,
                        'description': config['description']
                    })
        
        return available
    except FileNotFoundError:
        return []

def load_model_data(model_name):
    """
    Load all data for a specific model, handling naming inconsistencies.
    
    Args:
        model_name (str): Name of the model ('desert-hws', 'desert-lws', 'forest', 'rigolli')
    
    Returns:
        dict: Contains fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    data_dir = get_data_directory()
    model_dir = data_dir / config['directory']
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load heatmap data
    hmap_path = model_dir / config['hmap_file']
    if not hmap_path.exists():
        raise FileNotFoundError(f"Heatmap file not found: {hmap_path}")
    
    hmap_data = np.load(hmap_path)
    
    # Handle naming inconsistency
    hmap_key = config['hmap_key']
    if hmap_key not in hmap_data:
        # Fallback to other possible names
        possible_keys = ['fitted_p_heatmap', 'fitted_heatmap', 'heatmap', 'p_heatmap']
        for key in possible_keys:
            if key in hmap_data:
                hmap_key = key
                break
        else:
            raise KeyError(f"Could not find heatmap data in {hmap_path}. Available keys: {list(hmap_data.keys())}")
    
    # Load DataFrames
    whiff_path = model_dir / config['whiff_file'] 
    nowhiff_path = model_dir / config['nowhiff_file']
    
    if not whiff_path.exists():
        raise FileNotFoundError(f"Whiff data file not found: {whiff_path}")
    if not nowhiff_path.exists():
        raise FileNotFoundError(f"No-whiff data file not found: {nowhiff_path}")
    
    try:
        fdf = pd.read_hdf(whiff_path)
        fdf_nowhiff = pd.read_hdf(nowhiff_path)
    except Exception as e:
        raise RuntimeError(f"Error loading HDF5 files: {e}")
    
    return {
        'fitted_p_heatmap': hmap_data[hmap_key],
        'xedges': hmap_data['xedges'],
        'yedges': hmap_data['yedges'], 
        'fdf': fdf,
        'fdf_nowhiff': fdf_nowhiff
    }

def validate_model_data(data):
    """Validate that loaded model data has the expected structure."""
    required_keys = ['fitted_p_heatmap', 'xedges', 'yedges', 'fdf', 'fdf_nowhiff']
    
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required data key: {key}")
    
    # Validate DataFrame columns
    required_whiff_cols = ['avg_distance_along_streakline', 'avg_nearest_from_streakline', 
                          'mean_concentration', 'std_whiff', 'length_of_encounter', 'odor_intermittency']
    required_nowhiff_cols = ['avg_distance_along_streakline', 'avg_nearest_from_streakline',
                            'wc_nowhiff', 'wsd_nowhiff']
    
    for col in required_whiff_cols:
        if col not in data['fdf'].columns:
            raise ValueError(f"Missing required column in whiff data: {col}")
    
    for col in required_nowhiff_cols:
        if col not in data['fdf_nowhiff'].columns:
            raise ValueError(f"Missing required column in no-whiff data: {col}")
    
    return True