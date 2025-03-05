import pandas as pd

import statsmodels.tsa.forecasting.stl as tfc


class Forecasts:

    def __init__(self, data: pd.DataFrame, testing: pd.DataFrame, system: tfc.STLForecastResults, code: str):
        """

        :param data:
        :param testing:
        :param system: The results of the seasonal component model
        :param code: The identification code of an institution
        """

        self.__data = data
        self.__testing = testing
        self.__system = system
        self.__code = code

    def __estimates(self):
        pass

    def __tests(self):
        pass

    def __futures(self):
        pass

    def exc(self, ):
        pass
