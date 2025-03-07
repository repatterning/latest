
import logging

import pandas as pd
import numpy as np
import statsmodels.tsa.api as tap
import statsmodels.tsa.forecasting.stl as tfc

import src.modelling.sc.control
import src.elements.codes as ce


class Seasonal:

    def __init__(self, training: pd.DataFrame, arguments: dict, code: ce.Codes):
        """

        :param training: The data of an institution.
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__training = training
        self.__arguments = arguments
        self.__code = code

        # Seasonal Components Arguments
        self.__sc: dict = self.__arguments.get('sc')

        # Methods for estimating model parameters, and a covariance matrix calculation method
        self.__methods = ['cg', 'bfgs']
        self.__covariance = 'robust'

        # Controls
        self.__control = src.modelling.sc.control.Control()

    def __execute(self, architecture: tfc.STLForecast, method: str) \
            -> tfc.STLForecastResults | None:
        """

        :param architecture:
        :param method:
        :return:
        """

        return self.__control(
            architecture=architecture, method=method, covariance=self.__covariance)

    def __s_arima(self, method: str):
        """

        :param method:
        :return:
        """

        architecture = tfc.STLForecast(
            self.__training[['seasonal']],
            tap.SARIMAX,
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
            robust=True
        )

        logging.info('Try: Seasonal ARIMA (%s, %s)', method, self.__code.hospital_code)

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
