import typing
import pandas as pd

import src.modelling.splits


class Interface:

    def __init__(self, data: pd.DataFrame, arguments: dict):

        self.__data = data
        self.__arguments = arguments

        self.__splits = src.modelling.splits.Splits(arguments=self.__arguments)

    def __get_data(self, code: str) -> pd.DataFrame:

        return self.__data.copy().loc[self.__data['hospital_code'] == code, :]

    def __get_splits(self, data: pd.DataFrame):

        training, testing = self.__splits.exc(data=data.copy())

    def exc(self):
        """
        The testing data has <ahead> instances.  Altogether predict <2 * ahead> points
        into the future.  The first set of ahead points are for weekly evaluations of
        a week's model; the true value of the latter set of ahead points will be known
        in future.

        :return:
        """

        codes = self.__data['hospital_code'].unique()

        # DASK: computations = []
        for code in codes:
            """
            1. get data
            2. decompose
            3. split
            4. seasonal component modelling: naive
            5. trend component modelling: gaussian process
            6. overarching estimate
            """

            data = self.__get_data(code=code)
