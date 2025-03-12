"""Module interface.py"""

import src.elements.codes as ce
import src.elements.master as mr
import src.modelling.sc.algorithm
import src.modelling.sc.forecasts
import src.modelling.sc.page


class Interface:
    """
    The seasonal component modelling interface
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__arguments = arguments

    def exc(self, master: mr.Master, code: ce.Codes) -> mr.Master | None:
        """

        :param master: A named tuple consisting of an institutions training & testing data
        :param code: The health board & institution/hospital codes of an institution/hospital.<br>
        :return:
        """

        # The seasonal forecasting algorithms
        algorithm = src.modelling.sc.algorithm.Algorithm(arguments=self.__arguments)
        system = algorithm.exc(training=master.training, code=code)

        if system is None:
            return None

        # Next, extract forecasts/predictions and supplementary details, subsequently persist; via the
        # model's <page> & <forecasts>.
        src.modelling.sc.page.Page(system=system, code=code).exc()
        src.modelling.sc.forecasts.Forecasts(master=master, system=system).exc(
            arguments=self.__arguments, code=code)

        return master
