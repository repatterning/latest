"""Module splits.py"""
import typing

import dask
import pandas as pd

import config


class Splits:
    """
    The training & testing splits.
    """

    def __init__(self, data: pd.DataFrame, arguments: dict):
        """

        :param data: The data set consisting of the attendance numbers per institution/hospital.
        :param arguments: Modelling arguments.
        """

        self.__data = data.copy()
        self.__arguments = arguments

        # Instances
        self.__configurations = config.Config()

    @dask.delayed
    def __get_data(self, code: str) -> pd.DataFrame:
        """

        :param code: Hospital, institution, code.
        :return:
        """

        blob = self.__data.copy().loc[self.__data['hospital_code'] == code, :]
        blob.sort_values(by='week_ending_date', ascending=True, inplace=True)

        return blob

    @dask.delayed
    def __include(self, blob: pd.DataFrame) -> pd.DataFrame:
        """

        :param blob:
        :return:
        """

        return blob.copy()[:-self.__arguments.get('ahead')]

    @dask.delayed
    def __exclude(self, blob: pd.DataFrame) -> pd.DataFrame:
        """
        Excludes instances that will be predicted

        :param blob:
        :return:
        """

        return blob.copy()[-self.__arguments.get('ahead'):]

    def exc(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """

        :return:
        """

        codes = self.__data['hospital_code'].unique()

        # Splitting by institution
        computations = []
        for code in codes:
            blob = self.__get_data(code=code)
            include = self.__include(blob=blob)
            exclude = self.__exclude(blob=blob)
            computations.append([include, exclude])
        calculations = dask.compute(computations, scheduler='threads')[0]

        # Structure
        including = [calculations[i][0] for i in range(len(calculations))]
        excluding = [calculations[i][1] for i in range(len(calculations))]

        training = pd.concat(including, axis=0, ignore_index=True)
        testing = pd.concat(excluding, axis=0, ignore_index=True)

        return training, testing
