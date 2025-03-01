import pandas as pd

import statsmodels.tsa.seasonal as stsl


class Decompose:

    def __init__(self, arguments: dict):

        self.__arguments = arguments

    def exc(self, data: pd.DataFrame):

        res = stsl.STL(data['ln'], period=52, seasonal=51, trend_deg=1, seasonal_deg=0, robust=True).fit()
