import logging
import pandas as pd


class Interface:

    def __init__(self, arguments: dict):

        self.__arguments = arguments

    def exc(self, training: pd.DataFrame, testing: pd.DataFrame) -> bool:

        logging.info(self.__arguments)
        logging.info(training.head())
        logging.info(testing.head())

        return True
