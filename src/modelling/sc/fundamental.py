"""Module fundamental.py"""
import logging

import numpy as np
import pandas as pd
import statsmodels.tsa.arima.model as tar
import statsmodels.tsa.forecasting.stl

import src.elements.codes as ce
import src.modelling.sc.control


class Fundamental:
    """
    Focus: Autoregressive Integrated Moving Average (ARIMA)
    """

    def __init__(self, training: pd.DataFrame, arguments: dict, code: ce.Codes):
        """
        
        :param training: The data of an institution.<br>
        :param arguments: A set of model development, and supplementary, arguments.<br>
        :param code: The health board & institution/hospital codes of an institution/hospital.
        """

        self.__training = training
        self.__arguments = arguments
        self.__code = code

        # Seasonal Components Arguments
        self.__sc: dict = self.__arguments.get('sc')

        # Methods for estimating model parameters, and a covariance matrix calculation method
        self.__methods = ['statespace', 'innovations_mle']
        self.__covariance = 'robust'

        # Controls
        self.__control = src.modelling.sc.control.Control()

    def __execute(self, architecture: statsmodels.tsa.forecasting.stl.STLForecast, method: str)  \
            -> statsmodels.tsa.forecasting.stl.STLForecastResults | None:
        """
        
        :param architecture: The architecture underpinning the modelling step, i.e., the .fit() step.<br>
        :param method: A parameter estimation method
        :return: 
        """

        system = self.__control(
            architecture=architecture, method=method, covariance=self.__covariance, code=self.__code)

        return system

    def __arima(self, method: str):
        """
        
        :param method: A parameter estimation method
        :return: 
        """

        architecture = statsmodels.tsa.forecasting.stl.STLForecast(
            self.__training[['seasonal']],
            tar.ARIMA,
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
            robust=True)

        logging.info('Modelling: %s, ARIMA (method -> %s)', self.__code.hospital_code, method)

        return self.__execute(architecture=architecture,  method=method)

    def exc(self) -> statsmodels.tsa.forecasting.stl.STLForecastResults | None:
        """
        
        :return: 
        """

        state = None
        for i in np.arange(len(self.__methods)):
            if (state is None) & (i < len(self.__methods)):
                state = self.__arima(method=self.__methods[i])

        return state
