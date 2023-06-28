"""
conf module
This module contains the configuration files for the project in YAML format.
The constants.py file is used to load the configuration files and make them accessible to other modules.
"""

# Load the constants from the constants.py file
from .load import *

# Make the constants accessible to other modules
__all__ = dir(load)
