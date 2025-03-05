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

        algorithm = src.modelling.sc.algorithm.Algorithm(arguments=self.__arguments)
        system = algorithm.exc(data=master.training)
        
        src.modelling.sc.page.Page().exc(system=system, code=code.hospital_code)
        src.modelling.sc.forecasts.Forecasts(data=master.training, testing=master.testing, system=system).exc(
            arguments=self.__arguments, code=code)

        return True
