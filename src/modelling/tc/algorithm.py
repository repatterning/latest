
import pymc

import pandas as pd
import numpy as np


# noinspection PyTypeChecker
class Algorithm:

    def __init__(self, frames: pd.DataFrame) -> None:
        """
        
        :param frames:
        """

        self.__frames = frames
        self.__sequence = self.__frames['trend'].to_numpy()
        self.__indices = np.expand_dims(np.arange(self.__frames.shape[0]), axis=1)

    def exc(self):

        # Initialise the model
        with pymc.Model() as model_:
            pass

        with model_:

            # The data containers
            points = pymc.Data('points', self.__indices)
            observations = pymc.Data('observations', self.__sequence)

            # Specify a covariance function: https://docs.pymc.io/api/gp/cov.html
            # Initialise the parameters spatial_scaling, variance_control
            spatial_scaling = pymc.Gamma('spatial_scaling', alpha=2, beta=1)
            variance_control = pymc.HalfCauchy('variance_control', beta=5)
            cov = variance_control**2 * pymc.gp.cov.Matern52(input_dim=1, ls=spatial_scaling)

            # Specify the Gaussian Process (GP); the default mean function is `Zero`.
            gp_ = pymc.gp.Marginal(cov_func=cov)

            # Marginal Likelihood
            ml_sigma = pymc.HalfCauchy('ml_sigma', beta=5)
            gp_.marginal_likelihood('ml', X=points, y=observations, sigma=ml_sigma)

            # Inference:
            # pymc.sampling.jax.sample_blackjax_nuts(2000, chains=2, random_seed=5, target_accept=0.95)
            details_ = pymc.sample(draws=2000, tune=1000, chains=1, random_seed=5, nuts_sampler='numpyro', target_accept=0.95)

        return model_, gp_, details_
