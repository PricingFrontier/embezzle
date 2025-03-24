"""
Embezzle - A Python library for visualizing and building GLMs using statsmodels with PyQt

This library provides a user-friendly interface for creating, visualizing, and analyzing
Generalized Linear Models (GLMs) using statsmodels backend and PyQt for the GUI.
"""

__version__ = '0.1.0'

# Import key components to make them available at the package level
from embezzle.models import model_builder
from embezzle.ui import main_window