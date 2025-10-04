"""
COSMOS: A Data-Driven Probabilistic Time Series Simulator for Chemical Plumes

This package provides a simplified interface for creating realistic odor time series
simulations across different environments and scales.
"""

from .core import predictor, CosmosFast
from .data_loader import list_available_models

__version__ = "1.0.0"
__all__ = ["predictor", "CosmosFast", "list_available_models"]