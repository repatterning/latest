import logging

import numpy as np

import pymc
import arviz


class Forecasting:

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray):

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae

    def __execute(self, name: str, model_: pymc.model.Model, pred_noise: bool):

        with model_:
            self.__gp.conditional(name=name, Xnew=self.__abscissae, pred_noise=pred_noise)
            objects = pymc.sample_posterior_predictive(self.__details, var_names=[name])

        return model_, objects

    def exc(self, model: pymc.model.Model, execute_observation_noise_option: bool):
        """

        :param model:
        :param execute_observation_noise_option:
        :return:
        """


        model, predictions = self.__execute(name='estimates', model_=model, pred_noise=False)
        logging.info('Persist\n%s', predictions)

        if execute_observation_noise_option:
            model, n_predictions = self.__execute(name='n_estimates', model_=model, pred_noise=True)
            logging.info('Persist\n%s',n_predictions)

        # Persist
        logging.info(model)
        logging.info(self.__details)
