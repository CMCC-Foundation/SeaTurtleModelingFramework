import os

import pandas as pd

from src.conf import paths, variables
from src.lib.analysis import statistics
from src.lib.utility import logger as logManager


def analysis(logger=True):
    """Performs analysis on presence data.

    This function performs a t-test on presence data to compare the means of two groups: coastal and pelagic.
    The results of the t-test are saved to an excel file. The function also logs information about the analysis process if a logger is provided.

    Args:
        logger (bool or logging.Logger, optional): If True, a logger is set up using logManager.setup. If a logger object is provided, it is used for logging. Defaults to True.
    """
        
    if logger is True:
        logger = logManager.setup(paths.log.analysis)

    presence = pd.read_excel(paths.input.presence.excel)
    logger.debug("presence shape = %s", str(presence.shape))

    output = statistics.ttest_season(table=presence,
                            variables=variables.analysis,
                            class_column="Habitat",
                            classes=["coastal", "pelagic"],
                            get_greater=False,
                            get_percentage=True,
                            logger=logger)

    os.makedirs(paths.output.analysis.dir, exist_ok=True)
    output.to_excel(paths.output.analysis.ttest)