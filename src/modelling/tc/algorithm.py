"""Module algorithm.py"""
import logging
import os
import typing

import arviz
# noinspection PyUnresolvedReferences
import jax
import numpy as np
import pandas as pd
import pymc
import pymc.sampling.jax


class Algorithm:

    os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=constant_folding'

    def __init__(self, training: pd.DataFrame) -> None:
        """

        :param training: An institution's training data
        """

        # Data
        self.__training = training
        self.__sequence = self.__training['trend'].to_numpy()
        self.__indices = np.expand_dims(np.arange(self.__training.shape[0]), axis=1)

    @staticmethod
    def __chains(trend: dict) -> int:
        """
        Ensures the chains value is in line with processing units
        numbers, and computation logic.

        :param trend: The trend component node of the modelling & supplementary arguments
        :return:
        """

        if (trend.get('chain_method') == 'parallel') & (str(jax.local_devices()[0]).startswith('cuda')):
            return jax.device_count(backend='gpu')

        return trend.get('chains')

    # noinspection PyTypeChecker
    def exc(self, arguments: dict) -> typing.Tuple[pymc.model.Model, arviz.InferenceData, pd.DataFrame]:
        """

        :param arguments: A set of modelling & supplementary arguments
        :return:
        """

        trend: dict = arguments.get('tc')

        # Indices for forecasting beyond training data
        abscissae = np.arange(self.__training.shape[0] + (2 * arguments.get('ahead')))[:, None]

        with pymc.Model() as model_:
            """
            More about covariance function: https://docs.pymc.io/api/gp/cov.html
            """

            # The data containers
            points = pymc.Data('points', self.__indices)
            observations = pymc.Data('observations', self.__sequence)

            # Covariance function: Initialise the spatial scaling (ℓ) and variance control (η) parameters
            spatial_scaling = pymc.Gamma(
                'spatial_scaling',
                alpha=trend.get('covariance').get('spatial_scaling').get('alpha'),
                beta=trend.get('covariance').get('spatial_scaling').get('beta'))

            variance_control = pymc.HalfCauchy(
                'variance_control',
                beta=trend.get('covariance').get('variance_control').get('beta'))

            cov = variance_control**2 * pymc.gp.cov.Matern52(input_dim=1, ls=spatial_scaling)

            # Specify the Gaussian Process (GP); the default mean function is `Zero`.
            gp_ = pymc.gp.Marginal(cov_func=cov)

            # Marginal Likelihood
            ml_sigma = pymc.HalfCauchy('ml_sigma', beta=trend.get('ml_sigma').get('beta'))
            gp_.marginal_likelihood('ml', X=points, y=observations, sigma=ml_sigma)

            # Inference
            logging.info('CHAINS: %s', self.__chains(trend=trend))

            details_ = pymc.sample(
                draws=trend.get('draws'),
                tune=trend.get('tune'),
                chains=self.__chains(trend=trend),
                target_accept=trend.get('target_accept'),
                random_seed=arguments.get('seed'),
                nuts_sampler=trend.get('nuts_sampler'),
                nuts_sampler_kwargs={
                    'chain_method': trend.get('chain_method'),
                    'postprocessing_backend': arguments.get('device')}
            )

            mu, variance = gp_.predict(
                abscissae, point=arviz.extract(details_.get('posterior'), num_samples=1).squeeze(),
                diag=True, pred_noise=False)
            forecasts_ = pd.DataFrame(
                data={'abscissa': abscissae.squeeze(), 'mu': mu, 'std': np.sqrt(variance)})

        return model_, details_, forecasts_
