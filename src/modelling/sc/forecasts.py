import logging
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

        est: pd.DataFrame = self.__system.result.seasonal.to_frame()
        est.rename(columns={'season': 'seasonal_est'}, inplace=True)
        est = self.__data.copy()[['seasonal']].join(est.copy())
        logging.info(est)

    def __tests(self):
        pass

    def __futures(self):
        pass

    def exc(self, arguments: dict):
        """

        :param arguments:
        :return:
        """

        self.__system.forecast(steps=(2 * arguments.get('ahead'))).to_frame()
