"""Module algorithm.py"""
import logging
import os
import typing

import arviz
import jax
import numpy as np
import pandas as pd
import pymc
import pymc.sampling.jax


class Algorithm:

    os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=constant_folding'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['DP_INTRA_OP_PARALLELISM_THREADS'] = '8'
    os.environ['DP_INTER_OP_PARALLELISM_THREADS'] = '4'

    def __init__(self, training: pd.DataFrame) -> None:
        """

        :param training:
        """

        # Data
        self.__training = training
        self.__sequence = self.__training['trend'].to_numpy()
        self.__indices = np.expand_dims(np.arange(self.__training.shape[0]), axis=1)

    @staticmethod
    def __chains(tc: dict) -> int:
        """
        Ensures the chains value is in line with processing units
        numbers, and computation logic.

        :param tc:
        :return:
        """

        if (tc.get('chain_method') == 'parallel') & (str(jax.local_devices()[0]).startswith('cuda')):
            return jax.device_count(backend='gpu')

        return tc.get('chains')

    # noinspection PyTypeChecker
    def exc(self, arguments: dict) -> typing.Tuple[pymc.model.Model, arviz.InferenceData, pd.DataFrame]:
        """

        :return:
        """

        # Arguments
        tc: dict = arguments.get('tc')

        abscissae = np.arange(self.__training.shape[0] + (2 * arguments.get('ahead')))[:, None]

        with pymc.Model() as model_:

            # The data containers
            points = pymc.Data('points', self.__indices)
            observations = pymc.Data('observations', self.__sequence)

            # Specify a covariance function: https://docs.pymc.io/api/gp/cov.html
            # Initialise the spatial scaling (ℓ) and variance control (η) parameters
            spatial_scaling = pymc.Gamma(
                'spatial_scaling',
                alpha=tc.get('covariance').get('spatial_scaling').get('alpha'),
                beta=tc.get('covariance').get('spatial_scaling').get('beta'))
            variance_control = pymc.HalfCauchy(
                'variance_control',
                beta=tc.get('covariance').get('variance_control').get('beta'))
            cov = variance_control**2 * pymc.gp.cov.Matern52(input_dim=1, ls=spatial_scaling)

            # Specify the Gaussian Process (GP); the default mean function is `Zero`.
            gp_ = pymc.gp.Marginal(cov_func=cov)

            # Marginal Likelihood
            ml_sigma = pymc.HalfCauchy(
                'ml_sigma', beta=tc.get('ml_sigma').get('beta'))
            gp_.marginal_likelihood('ml', X=points, y=observations, sigma=ml_sigma)

            # Inference
            logging.info('CHAINS: %s', self.__chains(tc=tc))

            details_ = pymc.sample(
                draws=tc.get('draws'),
                tune=50, # self.__tc.get('tune'),
                chains=4, # self.__chains(),
                target_accept=tc.get('target_accept'),
                random_seed=arguments.get('seed'),
                nuts_sampler=tc.get('nuts_sampler'),
                nuts_sampler_kwargs={
                    'chain_method': tc.get('chain_method'),
                    'postprocessing_backend': arguments.get('device')}
            )

            mu, variance = gp_.predict(
                abscissae, point=arviz.extract(details_.get('posterior'), num_samples=1).squeeze(),
                diag=True, pred_noise=False)
            forecasts_ = pd.DataFrame(
                data={'abscissa': abscissae.squeeze(), 'mu': mu, 'std': np.sqrt(variance)})

        return model_, details_, forecasts_
