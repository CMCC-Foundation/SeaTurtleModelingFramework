"""analysis module
This module provides functions for performing statistical analysis on datasets.

The statistics.py file contains functions for performing statistical tests like the t-test.
"""

# Import the functions from the statistics.py file
from .statistics import ttest_season

# Make the functions accessible to other modules
__all__ = ['ttest_season']