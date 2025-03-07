"""Module interface.py"""
import logging

import src.elements.codes as ce
import src.elements.master as mr
import src.modelling.sc.algorithm
import src.modelling.sc.forecasts
import src.modelling.sc.page


class Interface:

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__arguments = arguments

    def exc(self, master: mr.Master, code: ce.Codes) -> bool:
        """

        :param master: A named tuple consisting of an institutions training & testing data
        :param code:
        :return:
        """

        # The seasonal forecasting algorithms
        algorithm = src.modelling.sc.algorithm.Algorithm(arguments=self.__arguments)
        system = algorithm.exc(training=master.training, code=code)

        if system is None:
            logging.info('Skipping: Seasonal forecasting for %s', code.hospital_code)
            return False

        # Extract, and persist, the model's details (page) and forecasts (forecasts).
        src.modelling.sc.page.Page(system=system, code=code).exc()
        src.modelling.sc.forecasts.Forecasts(master=master, system=system).exc(
            arguments=self.__arguments, code=code)
        logging.info('Latest: Seasonal forecasting of %s', code.hospital_code)

        return True
