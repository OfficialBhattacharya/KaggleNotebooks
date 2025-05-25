"""
DataModeller Package

This package provides tools for model comparison and data preprocessing.

Main functions:
- getModelReadyData: Preprocess data with various transformations
- fitPlotAndPredict: Fit models with cross-validation, generate plots, and make predictions
"""

from .getModelReadyData import getModelReadyData
from .modelCompare import fitPlotAndPredict

__all__ = ['getModelReadyData', 'fitPlotAndPredict'] 