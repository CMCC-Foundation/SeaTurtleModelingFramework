import inspect
import logging
import os
import pathlib
import sys


class StreamToLogger:
    """Stream object that redirects writes to a logger instance."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup(path, name=None, level=logging.INFO):
    """Sets up a logger.

    Args:
        path (str): Path to the directory where the log file will be stored.
        name (str, optional): Name of the logger. Defaults to the name of the calling file.
        level (int, optional): Logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The configured logger.

    """
    if name is None:
        name = pathlib.Path(inspect.stack()[1].filename).stem
    
    log_filename = "LOG_{v}.txt".format(v=name)

    os.makedirs(path, exist_ok=True)
    log_file = os.path.join(path, log_filename)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%d/%m/%Y %H:%M:%S")

    # Handler per il file di log
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger

def redirect_stdout(logger):
    """Redirects the standard output to a logger.

    This function redirects the standard output stream (sys.stdout) to a
    specified logger. All print statements will be redirected to the logger
    with a logging level of INFO.

    Args:
        logger: The logger object to which the standard output will be
            redirected.
    """
    sys.stdout = StreamToLogger(logger, logging.INFO)

def restore_stdout():
    """Restores the standard output to its original value.

    This function restores the standard output stream (sys.stdout) to its
    original value (sys.__stdout__).
    """
    sys.stdout = sys.__stdout__
