"""Module algorithm.py"""
import typing
import logging

import arviz
import jax
import numpy as np
import pandas as pd
import pymc
import pymc.sampling.jax


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

    def __chains(self, chain_method: str) -> int:
        """
        Ensures the chains value is in line with processing units
        numbers, and computation logic.

        :param chain_method:
        :return:
        """

        if (chain_method == 'parallel') & (str(jax.local_devices()[0]).startswith('cuda')):
            return jax.device_count(backend='gpu')

        return self.__tc.get('chains')

    # noinspection PyTypeChecker
    def exc(self) -> typing.Tuple[pymc.model.Model, pymc.gp.Marginal, arviz.InferenceData]:
        """

        :return:
        """

        with pymc.Model() as model_:

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

            logging.info('CHAINS: %s', self.__chains(chain_method='vectorized'))

            '''
            details_ = pymc.sampling.jax.sample_numpyro_nuts(
                draws=500, tune=150, chains=self.__chains(chain_method='vectorized'), 
                target_accept=0.95, random_seed=self.__arguments.get('seed'),
                chain_method='vectorized', postprocessing_backend='gpu'
            )
            '''

            details_ = pymc.sample(
                draws=500, # self.__tc.get('draws'),
                tune=150, # self.__tc.get('tune'),
                chains=self.__chains(chain_method='vectorized'),
                target_accept=self.__tc.get('target_accept'),
                random_seed=self.__arguments.get('seed'),
                nuts_sampler=self.__tc.get('nuts_sampler'),
                nuts_sampler_kwargs={'chain_method': 'vectorized', 'postprocessing_backend': 'gpu'}
            )

        return model_, gp_, details_
