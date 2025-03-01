import pandas as pd


class Interface:

    def __init__(self, training: pd.DataFrame):

        self.__training = training

    def __get_data(self, code: str) -> pd.DataFrame:

        return self.__training.copy().loc[self.__training['hospital_code'] == code, :]

    def exc(self):

        codes = self.__training['hospital_code'].unique()

        # DASK: computations = []
        for code in codes:
            """
            1. get data
            2. seasonal component modelling: naive
            3. trend component modelling: gaussian process
            4. overarching estimate + futures
            """

            data = self.__get_data(code=code)

