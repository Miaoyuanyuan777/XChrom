"""
XChrom - Cross cell chromatin accecebility prediction model

This package provides functions for preprocessing data, training XChrom model, 
caculating metrics for model performance and analyzing the model results.

Modules:
--------
- pp: Preprocessing data for model training and prediction
- tr: XChrom model architecture and training functions
- tl: Caculating evaluation metrics and analyzing the model results
- pl: Plotting functions for model results

"""

from . import tl, pp, tr, pl
from .readfile import *
from .data_access import *

__version__ = "v1.0.1"
__author__ = "Miao Yuanyuan"
__email__ = "miaoyuanyuan2022@sinh.ac.cn"