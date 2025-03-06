"""Module interface.py"""
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

    def exc(self, master: mr.Master, code: ce.Codes) -> str:
        """

        :param master: A named tuple consisting of an institutions training & testing data
        :param code:
        :return:
        """

        # The seasonal forecasting algorithm
        algorithm = src.modelling.sc.algorithm.Algorithm(arguments=self.__arguments)
        system = algorithm.exc(training=master.training)

        # Extract, and persist, the model's details (page) and forecasts (forecasts).
        src.modelling.sc.page.Page().exc(
            system=system, code=code)
        src.modelling.sc.forecasts.Forecasts(master=master, system=system).exc(
            arguments=self.__arguments, code=code)

        return f'Latest: Seasonal forecasting of {code.hospital_code}'
