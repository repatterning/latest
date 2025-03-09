"""Module forecasting.py"""
import os
import logging

import arviz
import numpy as np
import pymc

import config
import src.elements.codes as ce
import src.modelling.tc.page


class Forecasting:
    """
    <b>Notes</b><br>
    ------<br>

    Forecasts values for assessing the training phase and tests forecasts, and forecasts future values.<br>
    """

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray, code: ce.Codes):
        """

        :param gp: The model's gaussian process
        :param details: The inference data, vis-à-vis Bayesian modelling steps thus far.
        :param abscissae: A set of indices, in sequence, representing the (a) training data points, (b)
                          testing data points, and (c) points for future forecasts, i.e., beyond the
                          testing data points.
        :param code: The health board & institution/hospital codes of an institution/hospital.
        """

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae
        self.__code = code

        configurations = config.Config()
        self.__root = os.path.join(configurations.artefacts_, 'models', code.hospital_code)

    def __execute(self, name: str, model_: pymc.model.Model, pred_noise: bool):
        """

        :param name: A name for the prediction step
        :param model_: The model
        :param pred_noise: Should observation noise be included in predictions?
        :return:
        """

        with model_:
            self.__gp.conditional(name=name, Xnew=self.__abscissae, pred_noise=pred_noise)
            objects = pymc.sample_posterior_predictive(self.__details, var_names=[name])

        return model_, objects

    def __persist_inference_data(self, data: arviz.InferenceData, name: str):
        """

        :param data:
        :param name:
        :return:
        """

        pathstr = os.path.join(self.__root, f'{name}.nc')

        try:
            data.to_netcdf(filename=pathstr)
            logging.info('%s: %s', self.__code.hospital_code, os.path.basename(pathstr))
        except IOError as err:
            raise err from err

    def exc(self, model: pymc.model.Model):
        """

        :param model: The model
        :return:
        """

        model, predictions = self.__execute(name='estimates', model_=model, pred_noise=False)
        model, n_predictions = self.__execute(name='n_estimates', model_=model, pred_noise=True)

        # Persist: Inference Data
        for data, name in zip([self.__details, predictions, n_predictions], ['details', 'free', 'noisy']):
            self.__persist_inference_data(data=data, name=name)

        # Persist: Model
        src.modelling.tc.page.Page(
            model=model, code=self.__code).exc(label='model')
