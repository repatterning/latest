"""Module architecture.py"""
import collections
import logging
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.experimental.util as tfu
import tensorflow_probability.python.sts.components as tfc
import tf_keras

import src.elements.inference as ifr
import src.elements.master as mr
import src.functions.streams
import src.modelling.predicting


class Architecture:
    """
    Architecture
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

    def __model(self, _training: np.ndarray) -> tfc.Sum:
        """

        :param _training:
        :return:
        """

        month_of_year_effect = tfp.sts.Seasonal(
            num_seasons=self.__arguments.get('seasons').get('number_of'),
            num_steps_per_season=self.__arguments.get('seasons').get('steps_per'),
            observed_time_series=_training,
            name='month_of_year_effect')

        autoregressive = tfp.sts.Autoregressive(order=1, observed_time_series=_training, name='autoregressive')

        model = tfp.sts.Sum([month_of_year_effect, autoregressive],
                            observed_time_series=_training)

        return model

    def __variational(self, model, _training: np.ndarray) -> typing.Tuple[tfu.DeferredModule, pd.DataFrame]:
        """

        :param model:
        :param _training:
        :return:
        """

        posterior: tfu.DeferredModule = tfp.sts.build_factored_surrogate_posterior(
            model=model)

        # Evidence Lower Bound Loss Curve Data
        elb: tf.Tensor = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model.joint_distribution(observed_time_series=_training).log_prob,
            surrogate_posterior=posterior,
            optimizer=tf_keras.optimizers.Adam(learning_rate=self.__arguments.get('learning_rate')),
            num_steps=self.__arguments.get('n_variational_steps'),
            jit_compile=True,
            seed=self.__arguments.get('seed'),
            name='fit_surrogate_posterior')
        logging.info('Evidence Lower Bound: %s', type(elb))

        _elb = pd.DataFrame(data={
            'index': np.arange(elb.numpy().shape[0]),
            'evidence_lower_bound': elb.numpy()})

        return posterior, _elb

    def exc(self, master: mr.Master) -> ifr.Inference:
        """

        :param master:
        :return:
        """

        # Model
        model: tfc.Sum = self.__model(_training=master.training['measure'].values)

        # Posteriors & Evidence Lower Bound
        v_posterior, v_elb = self.__variational(model=model, _training=master.training['measure'].values)

        # Samples
        v_posterior_samples: collections.OrderedDict = v_posterior.sample(self.__arguments.get('n_samples'))

        # Hence
        v_estimates = src.modelling.predicting.Predicting(arguments=self.__arguments).exc(
            master=master, model=model, v_posterior_samples=v_posterior_samples)

        return ifr.Inference(evidence_lower_bound=v_elb, estimates=v_estimates)
