"""Module interface.py"""
import logging

import numpy as np
import pandas as pd

import src.elements.codes as ce
import src.modelling.tc.algorithm
import src.modelling.tc.forecasting


class Interface:

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

    def exc(self, training: pd.DataFrame, code: ce.Codes, state: bool) -> str:
        """

        :param training: The training data of an institution.
        :param code: The health board & institution/hospital codes of an institution/hospital.
        :param state:
        :return:
        """

        if state:

            # Model, etc.
            model, gp, details  = src.modelling.tc.algorithm.Algorithm(
                training=training, arguments=self.__arguments).exc()

            # Estimates & Futures
            abscissae = np.arange(
                training.shape[0] + (2 * self.__arguments.get('ahead'))
            )[:, None]
            logging.info('%s\n%s', abscissae.shape, abscissae)
            src.modelling.tc.forecasting.Forecasting(
                gp=gp, details=details, abscissae=abscissae, code=code).exc(model=model)

            return f'Trend Component Modelling: Success -> {code.hospital_code}'

        return f'Skip: {code.hospital_code}'
