import pandas as pd


class Interface:

    def __init__(self, training: pd.DataFrame):

        self.__training = training

    def __get_data(self, code: str) -> pd.DataFrame:

        return self.__training.copy().loc[self.__training['hospital_code'] == code, :]

    def exc(self):
        """
        The testing data has <ahead> instances.  Altogether predict <2 * ahead> points
        into the future.  The first set of ahead points are for weekly evaluations of
        a week's model; the true value of the latter set of ahead points will be known
        in future.

        :return:
        """

        codes = self.__training['hospital_code'].unique()

        # DASK: computations = []
        for code in codes:
            """
            1. get data
            2. decompose
            3. seasonal component modelling: naive
            4. trend component modelling: gaussian process
            5. overarching estimate + futures
            """

            data = self.__get_data(code=code)
