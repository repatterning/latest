
import pandas as pd

import statsmodels.tsa.forecasting.stl as tfc
import statsmodels.tsa.arima.model as tar


class Algorithm:

    def __init__(self, arguments: dict):

        self.__arguments: dict = arguments
        self.__sc: dict = self.__arguments.get('sc')

    def exc(self, data: pd.DataFrame):
        """
        <b>References</b><br>
        <a href="https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html">
        ARIMA (Autoregressive Integrated Moving Average)</a><br>
        <a href="https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html">STLForecast</a><br><br>

        :param data: The data of an institution, including the decompositions of its <i>ln(attendance numbers)</i> series.
        :return:
        """

        data.index.freq = self.__arguments.get('frequency')

        tfc.STLForecast(
            data[['seasonal']], tar.ARIMA,
            model_kwargs=dict(
                seasonal_order=(
                    self.__sc.get('P'),
                    self.__sc.get('D'),
                    self.__sc.get('Q'),
                    self.__sc.get('m')),
                trend='c'),
            seasonal=self.__sc.get('smoother_seasonal'),
            seasonal_deg=self.__sc.get('degree_seasonal'),
            trend_deg=self.__sc.get('degree_trend'),
            robust=False)
