"""Module interface.py"""
import logging
import typing

import arviz
import numpy as np
import pandas as pd
import pymc

import src.elements.codes as ce
import src.modelling.tc.algorithm
import src.modelling.tc.forecasting
import src.modelling.tc.page


class Interface:

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

    def __trend_component_modelling(self, training: pd.DataFrame) -> (
            typing.Tuple)[pymc.model.Model, pymc.gp.Marginal, arviz.InferenceData]:
        """

        :param training: The training data of an institution.
        :return:
        """

        algorithm = src.modelling.tc.algorithm.Algorithm(training=training, arguments=self.__arguments)

        return algorithm.exc()

    def exc(self, training: pd.DataFrame, code: ce.Codes, state: bool) -> str:
        """

        :param training: The training data of an institution.
        :param code: The health board & institution/hospital codes of an institution/hospital.
        :param state: If the seasonal component modelling step was a success, this will be True; otherwise False.
        :return:
        """

        if not state:
            return f'Trend Component Modelling: Skip -> {code.hospital_code}'

        # Model
        model, gp, details = self.__trend_component_modelling(training=training)

        # Persist: Algorithm
        src.modelling.tc.page.Page(model=model, code=code).exc(label='algorithm')

        # Estimates & Futures
        abscissae = np.arange(
            training.shape[0] + (2 * self.__arguments.get('ahead'))
        )[:, None]
        src.modelling.tc.forecasting.Forecasting(
            gp=gp, details=details, abscissae=abscissae, code=code).exc(model=model)

        return f'Trend Component Modelling: Success -> {code.hospital_code}'
