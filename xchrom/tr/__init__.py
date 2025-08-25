"""
Training functions and models for XChrom.

This module contains the core training components including:
- XChrom model architecture and layers
- Data generators for training
- Training utilities and callbacks

"""

from .train import (train_XChrom)
from ._utils import (
    Generator,
    XChrom_model
)

__all__ = [
    'train_XChrom',
    'Generator', 
    'XChrom_model'
]
