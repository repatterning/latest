"""Module fundamental.py"""
import logging

import numpy as np
import pandas as pd
import statsmodels.tsa.arima.model as tar
import statsmodels.tsa.forecasting.stl as tfc

import src.modelling.sc.control


class Fundamental:

    def __init__(self, training: pd.DataFrame, arguments: dict):
        """
        
        :param training: The data of an institution.
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__training = training
        self.__arguments = arguments

        # Seasonal Components Arguments
        self.__sc: dict = self.__arguments.get('sc')

        # Parameters estimation methods, and a covariance matrix calculation method
        self.__methods = ['statespace', 'innovations_mle']
        self.__covariance = 'robust'

        # Controls
        self.__control = src.modelling.sc.control.Control()

    def __execute(self, architecture: tfc.STLForecast, method: str)  -> tfc.STLForecastResults | None:
        """
        issue = issubclass(el[-1].category, sme.ConvergenceWarning)
        
        :param architecture:
        :param method: A parameter estimation method
        :return: 
        """

        system = self.__control(
            architecture=architecture, method=method, covariance=self.__covariance)

        return system

    def __arima(self, method: str):
        """
        
        :param method: 
        :return: 
        """

        architecture = tfc.STLForecast(
            self.__training[['seasonal']], tar.ARIMA,
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

        logging.info(f'Try: ARIMA, %s', method)

        return self.__execute(architecture=architecture,  method=method)

    def exc(self) -> tfc.STLForecastResults | None:
        """
        
        :return: 
        """

        state = None
        for i in np.arange(len(self.__methods)):
            if (state is None) & (i < len(self.__methods)):
                state = self.__arima(method=self.__methods[i])

        return state
