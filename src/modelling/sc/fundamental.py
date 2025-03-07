import logging
import warnings

import pandas as pd
import numpy as np

import statsmodels.tsa.forecasting.stl as tfc
import statsmodels.tsa.arima.model as tar
import statsmodels.tools.sm_exceptions as sme


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

        # Parameters estimation methods
        self.__methods = ['statespace', 'innovations_mle']

        # Covariance matrix calculation
        self.__covariance = 'robust'

    def __execute(self, arima: tfc.STLForecast, method: str)  -> tfc.STLForecastResults | None:
        """
        issue = issubclass(el[-1].category, sme.ConvergenceWarning)
        
        :param arima: 
        :param method: A parameter estimation method
        :return: 
        """

        with warnings.catch_warnings(record=True) as el:

            warnings.simplefilter('always')
            warnings.warn('Convergence', category=sme.ConvergenceWarning)
            
            system = arima.fit(fit_kwargs={'method': method, 'cov_type': self.__covariance})

            query = str(el[-1].message).__contains__('failed to converge')
            warnings.resetwarnings()

        if query:
            return None

        return system

    def __arima(self, method: str):
        """
        
        :param method: 
        :return: 
        """

        arima = tfc.STLForecast(
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

        logging.info(f'Try: ARIMA, {method}')

        return self.__execute(arima=arima,  method=method)

    def exc(self) -> tfc.STLForecastResults | None:
        """
        
        :return: 
        """

        state = None
        for i in np.arange(len(self.__methods)):
            if (state is None) & (i < len(self.__methods)):
                state = self.__arima(method=self.__methods[i])

        return state
