
import pymc

import pandas as pd
import numpy as np


# noinspection PyTypeChecker
class Algorithm:

    def __init__(self, training: pd.DataFrame, arguments: dict) -> None:
        """

        :param training:
        :param arguments:
        """

        # Data
        self.__training = training
        self.__sequence = self.__training['trend'].to_numpy()
        self.__indices = np.expand_dims(np.arange(self.__training.shape[0]), axis=1)

        # Arguments
        self.__arguments = arguments
        self.__tc: dict = arguments.get('tc')

    def exc(self):

        # Initialise the model
        with pymc.Model() as model_:
            pass

        with model_:

            # The data containers
            points = pymc.Data('points', self.__indices)
            observations = pymc.Data('observations', self.__sequence)

            # Specify a covariance function: https://docs.pymc.io/api/gp/cov.html
            # Initialise the spatial scaling (ℓ) and variance control (η) parameters
            spatial_scaling = pymc.Gamma(
                'spatial_scaling',
                alpha=self.__tc.get('covariance').get('spatial_scaling').get('alpha'),
                beta=self.__tc.get('covariance').get('spatial_scaling').get('beta'))
            variance_control = pymc.HalfCauchy(
                'variance_control',
                beta=self.__tc.get('covariance').get('variance_control').get('beta'))
            cov = variance_control**2 * pymc.gp.cov.Matern52(input_dim=1, ls=spatial_scaling)

            # Specify the Gaussian Process (GP); the default mean function is `Zero`.
            gp_ = pymc.gp.Marginal(cov_func=cov)

            # Marginal Likelihood
            ml_sigma = pymc.HalfCauchy(
                'ml_sigma', beta=self.__tc.get('ml_sigma').get('beta'))
            gp_.marginal_likelihood('ml', X=points, y=observations, sigma=ml_sigma)

            # Inference
            details_ = pymc.sample(
                draws=self.__tc.get('draws'),
                tune=self.__tc.get('tune'),
                chains=self.__tc.get('chains'),
                random_seed=self.__arguments.get('seed'),
                nuts_sampler=self.__tc.get('nuts_sampler'),
                target_accept=self.__tc.get('target_accept'))

        return model_, gp_, details_
