"""Module core.py"""
import dask

import src.elements.master as mr
import src.functions.streams
import src.modelling.tc.interface


class Core:
    """
    Trend component modelling via parallel computation; re-visit.
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__arguments = arguments

    def exc(self, masters: list[mr.Master]) -> list[str]:
        """

        :param masters: A named tuple consisting of an institutions training & testing data
        :return:
        """

        tc = dask.delayed(src.modelling.tc.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for master in masters:
            message = tc(training=master.training)
            computations.append(message)
        messages = dask.compute(computations, scheduler='threads')[0]

        return messages
