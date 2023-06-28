"""This module contains functions for preprocessing, training, and analyzing data.

The `preprocessing` function preprocesses presence and absence data by generating presence and absence data, cutting the data, and extracting features from the data.

The `training` function trains a classifier using presence and absence data. The classifier used is a Random Forest Classifier and the search method is BayesSearchCV.

The `analysis` function performs a t-test on presence data to compare the means of two groups: coastal and pelagic. The results of the t-test are saved to an excel file.

All functions also log information about their respective processes if a logger is provided.
"""

# from .preprocessing import preprocessing
from .train import training
from .analysis import analysis

# Make the functions accessible to other modules
__all__ = ['training', 'analysis']