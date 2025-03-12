"""Module control.py"""
import logging
import warnings

import statsmodels.tools.sm_exceptions as sme
import statsmodels.tsa.forecasting.stl as tfc

import src.elements.codes as ce


class Control:
    """
    A warnings control system.
    """

    def __init__(self):
        pass

    def __call__(self, architecture: tfc.STLForecast, method: str, covariance: str, code: ce.Codes) \
            -> tfc.STLForecastResults | None:
        """
        issue = issubclass(el[-1].category, sme.ConvergenceWarning)

        :param architecture: The architecture underpinning the modelling step, i.e., the .fit() step.<br>
        :param method: The parameter estimation method, vis-à-vis
            <a href="www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html">ARIMA</a>,
            <a href="www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html">
            Seasonal ARIMA</a>.
        :param covariance: The covariance calculation method, vis-à-vis
            <a href="www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html">ARIMA</a>,
            <a href="www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html">
            Seasonal ARIMA</a>.
        :param code: The health board & institution/hospital codes of an institution/hospital.<br>
        :return:
        """

        with warnings.catch_warnings(record=True) as el:

            warnings.simplefilter('always')
            warnings.warn('Convergence', category=sme.ConvergenceWarning)

            system = architecture.fit(fit_kwargs={'method': method, 'cov_type': covariance})

            query = (str(el[-1].message).__contains__('failed to converge') |
                     str(el[-1].message).__contains__('did not converge'))
            warnings.resetwarnings()

        if query:
            logging.info('Skip: %s, %s (method -> %s)', code.hospital_code, architecture.__getattribute__('_model'), method)
            return None

        system.__setattr__('parameters_estimation_method', method)

        return system
