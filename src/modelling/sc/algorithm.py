"""Module algorithm.py"""
import logging

import pandas as pd
import statsmodels.tsa.forecasting.stl as tfc

import src.elements.codes as ce
import src.modelling.sc.fundamental
import src.modelling.sc.seasonal


class Algorithm:
    """
    Class Algorithm
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments: dict = arguments

    def exc(self, training: pd.DataFrame, code: ce.Codes) -> tfc.STLForecastResults | None:
        """
        <b>References</b><br>
        <a href="https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html">
        ARIMA (Autoregressive Integrated Moving Average)</a><br>
        <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html">
        SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous)</a><br>
        <a href="https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html">
        STLForecast</a><br><br>

        :param training: The data of an institution, including the decompositions of its <i>ln(attendance numbers)</i> series.<br>
        :param code: The health board & institution/hospital codes of an institution/hospital.<br>
        :return:
        """

        logging.info('Modelling %s', code.hospital_code)

        # Setting data frequency
        training.index.freq = self.__arguments.get('frequency')

        # Modelling
        system: tfc.STLForecastResults = src.modelling.sc.fundamental.Fundamental(
            training=training, arguments=self.__arguments).exc()

        if system is None:
            system: tfc.STLForecastResults = src.modelling.sc.seasonal.Seasonal(
                training=training, arguments=self.__arguments).exc()

        return system
