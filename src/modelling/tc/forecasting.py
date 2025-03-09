import os

import arviz
import numpy as np
import pymc

import config
import src.elements.codes as ce


class Forecasting:

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray, code: ce.Codes):
        """

        :param gp:
        :param details:
        :param abscissae:
        :param code:
        """

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae
        self.__code = code

        configurations = config.Config()
        self.__root = os.path.join(configurations.artefacts_, 'models', code.hospital_code)

    def __execute(self, name: str, model_: pymc.model.Model, pred_noise: bool):
        """

        :param name:
        :param model_:
        :param pred_noise:
        :return:
        """

        with model_:
            self.__gp.conditional(name=name, Xnew=self.__abscissae, pred_noise=pred_noise)
            objects = pymc.sample_posterior_predictive(self.__details, var_names=[name])

        return model_, objects

    def __inferences(self, data: arviz.InferenceData, name: str):
        """

        :param data:
        :param name:
        :return:
        """

        pathstr = os.path.join(self.__root, f'{name}.nc')

        try:
            data.to_netcdf(filename=pathstr)
        except IOError as err:
            raise err from err

    def exc(self, model: pymc.model.Model):
        """

        :param model: The model
        :return:
        """

        model, predictions = self.__execute(name='estimates', model_=model, pred_noise=False)
        model, n_predictions = self.__execute(name='n_estimates', model_=model, pred_noise=True)

        # Persist
        for data, name in zip([self.__details, predictions, n_predictions], ['details', 'free', 'noisy']):
            self.__inferences(data=data, name=name)
