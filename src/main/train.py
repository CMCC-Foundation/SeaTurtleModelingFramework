import os
import pandas as pd

from src.lib.ml import model
from src.conf import paths, variables
from src.lib.utility import logger as logManager
from sklearn.ensemble import RandomForestClassifier

def training(logger=True):
    """Trains a classifier using presence and absence data.

    This function trains a classifier using presence and absence data from excel files.
    The classifier used is a Random Forest Classifier and the search method is BayesSearchCV.
    The function also logs information about the training process if a logger is provided.

    Args:
        logger (bool or logging.Logger, optional): If True, a logger is set up using logManager.setup. 
            If a logger object is provided, it is used for logging. Defaults to True.

    Returns:
        dict: A dictionary containing information about the training output.
    """
    
    # Set up logger if logger is True
    if logger is True:
        logger = logManager.setup(paths.log.ml)

    # Read presence and absence data from excel files
    presence = pd.read_excel(paths.input.presence.excel)
    absence = pd.read_excel(paths.input.absence.excel)

    # Add status column to presence and absence data
    presence["Status"] = "Presence"
    absence["Status"] = "Absence"

    # Concatenate presence and absence data
    turtles = pd.concat([absence, presence], axis=0, ignore_index=True)
    # Get features for learning 
    features = [col for col in turtles for var in variables.learning if col.startswith(var)]

    logger.debug("shape turtles = %s", str(turtles.shape)) if logger is not None else None
    logger.debug("num features = %s", str(len(features))) if logger is not None else None
    logger.debug("variables learning = %s", variables.learning) if logger is not None else None

    # Get X and y data for training and concanate them to create the dataset
    X = turtles[features]
    y = turtles[["Status"]]
    dataset = pd.concat([X, y], axis=1)

    logger.debug("X shape = %s", str(X.shape)) if logger is not None else None
    logger.debug("y shape = %s", str(y.shape)) if logger is not None else None
    logger.debug("total dataset shape = %s", str(dataset.shape)) if logger is not None else None

    # Train classifier
    training_output = model.optimize(
        dataset=dataset,
        exp_name="HABITAT",
        estimator=RandomForestClassifier(),
        search="BayesSearchCV",
        logger=logger,
        outdir=paths.output.training,
    )
    
    logger.info("Training output: %s", str(training_output)) if logger is not None else None