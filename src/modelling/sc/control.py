import warnings

import statsmodels.tools.sm_exceptions as sme
import statsmodels.tsa.forecasting.stl as tfc


class Control:

    def __init__(self):
        pass

    def __call__(self, architecture: tfc.STLForecast, method: str, covariance: str) -> tfc.STLForecastResults | None:

        with warnings.catch_warnings(record=True) as el:

            warnings.simplefilter('always')
            warnings.warn('Convergence', category=sme.ConvergenceWarning)

            system = architecture.fit(fit_kwargs={'method': method, 'cov_type': covariance})

            query = str(el[-1].message).__contains__('failed to converge')
            warnings.resetwarnings()

        if query:

            return None

        return system
