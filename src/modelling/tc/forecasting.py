"""Module forecasting.py"""
import logging
import os

import arviz
import numpy as np
import pandas as pd
import pymc

import config
import src.functions.streams
import src.modelling.tc.page


class Forecasting:
    """
    <b>Notes</b><br>
    ------<br>

    Forecasts values for assessing the training phase and tests forecasts, and forecasts future values.<br>
    """

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray, institution: str):
        """

        :param gp: The model's gaussian process
        :param details: The inference data, vis-à-vis Bayesian modelling steps thus far.
        :param abscissae: A set of indices, in sequence, representing the (a) training data points, (b)
                          testing data points, and (c) points for future forecasts, i.e., beyond the
                          testing data points.
        :param institution: An institution/hospital code.
        """

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae
        self.__institution = institution

        configurations = config.Config()
        self.__path = os.path.join(configurations.artefacts_, 'models', self.__institution)

    def __execute(self, model_: pymc.model.Model, pred_noise: bool) -> pd.DataFrame:
        """

        :param model_: The model
        :param pred_noise: Should observation noise be included in predictions?
        :return:
        """

        with model_:
            mu, variance = self.__gp.predict(
                self.__abscissae,
                point=arviz.extract(self.__details.get('posterior'), num_samples=1).squeeze(),
                diag=True, pred_noise=pred_noise)

        return pd.DataFrame(
            data={'abscissa': self.__abscissae.squeeze(), 'mu': mu, 'std': np.sqrt(variance)})

    def __persist_inference_data(self, data: arviz.InferenceData, name: str):
        """

        :param data:
        :param name:
        :return:
        """

        pathstr = os.path.join(self.__path, f'{name}.nc')

        try:
            data.to_netcdf(filename=pathstr)
            logging.info('%s: %s', self.__institution, os.path.basename(pathstr))
        except IOError as err:
            raise err from err

    def exc(self, model: pymc.model.Model):
        """

        :param model: The model
        :return:
        """

        # Persist: Model
        src.modelling.tc.page.Page(
            model=model, path=self.__path).exc(label='algorithm')

        # Persist: Inference Data
        self.__persist_inference_data(data=self.__details, name='tcf_details')

        # Forecasts
        forecasts = self.__execute(model_=model, pred_noise=False)
        src.functions.streams.Streams().write(
            blob=forecasts, path=os.path.join(self.__path, 'tcf_forecasts.csv'))
