import pandas as pd

import statsmodels.tsa.seasonal as stsl


class Decompose:

    def __init__(self, arguments: dict):

        self.__arguments: dict = arguments
        self.__decompose: dict = self.__arguments.get('decompose')

    def exc(self, data: pd.DataFrame):

        components = stsl.STL(data['ln'], period=self.__arguments.get('seasons'),
                              seasonal=self.__decompose.get('smoother_seasonal'),
                              trend_deg=self.__decompose.get('degree_trend'),
                              seasonal_deg=self.__decompose.get('degree_seasonal'),
                              robust=True).fit()
