"""Module seasonal.py"""
import logging

import numpy as np
import pandas as pd
import statsmodels.tsa.api as tap
import statsmodels.tsa.forecasting.stl as tfc

import src.elements.codes as ce
import src.modelling.sc.control


class Seasonal:
    """
    Focus: Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    """

    def __init__(self, training: pd.DataFrame, arguments: dict, code: ce.Codes):
        """

        :param training: The data of an institution.
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__training = training
        self.__arguments = arguments
        self.__code = code

        # Seasonal Components Arguments
        self.__sc_arguments: dict = self.__arguments.get('sc')

        # Methods for estimating model parameters, and a covariance matrix calculation method
        self.__methods = ['cg', 'bfgs']
        self.__covariance = 'robust'

        # Controls
        self.__control = src.modelling.sc.control.Control()

    def __execute(self, architecture: tfc.STLForecast, method: str) \
            -> tfc.STLForecastResults | None:
        """

        :param architecture: The architecture underpinning the modelling step, i.e., the .fit() step.<br>
        :param method: A parameter estimation method
        :return:
        """

        return self.__control(
            architecture=architecture, method=method, covariance=self.__covariance, code=self.__code)

    def __s_arima(self, method: str):
        """

        :param method: A parameter estimation method
        :return:
        """

        architecture = tfc.STLForecast(
            self.__training[['seasonal']],
            tap.SARIMAX,
            model_kwargs={
                "seasonal_order": (
                    self.__sc_arguments.get('P'),
                    self.__sc_arguments.get('D'),
                    self.__sc_arguments.get('Q'),
                    self.__sc_arguments.get('m')),
                "trend": "c"},
            seasonal=self.__sc_arguments.get('smoother_seasonal'),
            seasonal_deg=self.__sc_arguments.get('degree_seasonal'),
            trend_deg=self.__sc_arguments.get('degree_trend'),
            robust=True
        )

        logging.info('Modelling: %s, Seasonal ARIMA (method -> %s)', self.__code.hospital_code, method)

        return self.__execute(architecture=architecture, method=method)

    def exc(self) -> tfc.STLForecastResults | None:
        """

        :return:
        """

        state = None

        for i in np.arange(len(self.__methods)):
            if (state is None) & (i < len(self.__methods)):
                state = self.__s_arima(method=self.__methods[i])

        return state
