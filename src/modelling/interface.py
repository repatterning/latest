"""Module interface.py"""

import pandas as pd

import config
import src.functions.directories
import src.modelling.codes
import src.modelling.core


class Interface:
    """
    The interface to the seasonal & trend component modelling steps.
    """

    def __init__(self, data: pd.DataFrame, arguments: dict):
        """

        :param data: For reading ...
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__data = data
        self.__arguments = arguments

        # Instances
        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()

    def exc(self):
        """
        Via dask dataframe, read-in a machine's set of measures files.  Subsequently, split, then add the
        relevant features to the training data split.

        :return:
        """
