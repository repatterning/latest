"""Module algorithm.py"""
import logging

import numpy as np
import pandas as pd
import statsmodels.tsa.arima.model as tar
import statsmodels.tsa.forecasting.stl as tfs

import src.elements.gauge as ge
import src.modelling.architecture.control


class Algorithm:
    """
    Focus: Autoregressive Integrated Moving Average (ARIMA)
    """

    def __init__(self, training: pd.DataFrame, arguments: dict, gauge: ge.Gauge):
        """
        
        :param training: The data vis-à-vis a gauge.<br>
        :param arguments: A set of model development, and supplementary, arguments.<br>
        :param gauge: Encodes the time series & catchment identification codes of a gauge, and its gauge datum.<br>
        """

        self.__training = training
        self.__arguments = arguments
        self.__gauge = gauge

        # Modelling Parameters
        self.__parameters: dict = self.__arguments.get('parameters')

        # Methods for estimating model parameters, and a covariance matrix calculation method
        self.__methods = ['statespace', 'innovations_mle']
        self.__covariance = 'robust'

        # Controls
        self.__control = src.modelling.architecture.control.Control()

    def __get_sequence(self) -> pd.Series:

        frame = self.__training.copy()[['date', 'measure']]
        frame.set_index(keys='date', inplace=True)
        frame.index.freq = self.__arguments.get('frequency')

        return frame['measure']

    def __execute(self, architecture: tfs.STLForecast, method: str)  \
            -> tfs.STLForecastResults | None:
        """
        
        :param architecture: The architecture underpinning the modelling step, i.e., the .fit() step.<br>
        :param method: A parameter estimation method
        :return: 
        """

        system = self.__control(
            architecture=architecture, method=method, covariance=self.__covariance, gauge=self.__gauge)

        return system

    def __arima(self, method: str):
        """
        
        :param method: A parameter estimation method
        :return: 
        """

        architecture = tfs.STLForecast(
            self.__get_sequence(),
            tar.ARIMA,
            model_kwargs={
                "order": (self.__parameters.get('p'), self.__parameters.get('d'), self.__parameters.get('q')),
                "trend": "t"},
            seasonal_deg=self.__parameters.get('degree_seasonal'),
            trend_deg=self.__parameters.get('degree_trend'),
            robust=self.__parameters.get('robust'))

        logging.info('Modelling: %s of %s, ARIMA (method -> %s)', self.__gauge.ts_id, self.__gauge.catchment_id, method)

        return self.__execute(architecture=architecture,  method=method)

    def exc(self) -> tfs.STLForecastResults | None:
        """
        
        :return: 
        """

        state = None
        for i in np.arange(len(self.__methods)):
            if (state is None) & (i < len(self.__methods)):
                state = self.__arima(method=self.__methods[i])

        return state
