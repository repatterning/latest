import typing
import pandas as pd
import dask

import src.modelling.splits
import src.modelling.decompose


class Interface:

    def __init__(self, data: pd.DataFrame, arguments: dict):

        self.__data = data
        self.__arguments = arguments

    @dask.delayed
    def __get_data(self, code: str) -> pd.DataFrame:

        return self.__data.copy().loc[self.__data['hospital_code'] == code, :]

    def exc(self):
        """
        The testing data has <ahead> instances.  Altogether predict <2 * ahead> points
        into the future.  The first set of ahead points are for weekly evaluations of
        a week's model; the true value of the latter set of ahead points will be known
        in future.

        :return:
        """

        codes = self.__data['hospital_code'].unique()

        # Additional delayed tasks
        decompose = dask.delayed(src.modelling.decompose.Decompose(arguments=self.__arguments).exc)
        splitting = dask.delayed(src.modelling.splits.Splits(arguments=self.__arguments).exc)

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
            decompositions = decompose(data=data)
            training, testing = splitting(data=decompositions)
