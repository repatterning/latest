"""Module control.py"""
import logging
import warnings

import statsmodels.tools.sm_exceptions as sme
import statsmodels.tsa.forecasting.stl as tfc

import src.elements.gauge as ge


class Control:
    """
    A warnings control system.
    """

    def __init__(self):
        pass

    def __call__(self, architecture: tfc.STLForecast, method: str, covariance: str, gauge: ge.Gauge) \
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
        :param gauge: Encodes the time series & catchment identification codes of a gauge.<br>
        :return:
        """

        with warnings.catch_warnings(record=True) as el:

            warnings.simplefilter('always')
            warnings.warn('Convergence', category=sme.ConvergenceWarning)

            system = architecture.fit(fit_kwargs={'method': method, 'cov_type': covariance})

            query = (str(el[-1].message).__contains__('failed to converge') |
                     str(el[-1].message).__contains__('did not converge') |
                     str(el[-1].message).__contains__('error not necessarily achieved'))

            if query:
                logging.info('Skip: %s (method -> %s), vis-à-vis %s of %s',
                             architecture.__getattribute__('_model'), method, gauge.ts_id, gauge.catchment_id)
                warnings.resetwarnings()
                return None

        warnings.resetwarnings()

        system.__setattr__('parameters_estimation_method', method)

        return system
