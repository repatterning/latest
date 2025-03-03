"""Module splits.py"""
import typing

import pandas as pd


class Splits:
    """
    The training & testing splits.
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: Modelling arguments.
        """

        self.__arguments = arguments

    def __include(self, blob: pd.DataFrame) -> pd.DataFrame:
        """

        :param blob:
        :return:
        """

        return blob.copy()[:-self.__arguments.get('ahead')]

    def __exclude(self, blob: pd.DataFrame) -> pd.DataFrame:
        """
        Excludes instances that will be predicted

        :param blob:
        :return:
        """

        return blob.copy()[-self.__arguments.get('ahead'):]

    def exc(self, data: pd.DataFrame,) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """

        :param data: The data set consisting of the attendance numbers of <b>an</b> institution/hospital.
        :return:
        """

        training = self.__include(blob=data)
        testing = self.__exclude(blob=data)

        return training, testing
