
import glob
import os

import dask

import config
import src.elements.codes as ce
import src.elements.master as mr
import src.functions.streams
import src.modelling.tc.interface


class Core:
    """
    
    """

    def __init__(self, codes: list[ce.Codes], arguments: dict):
        """

        :param codes:
        :param arguments:
        """

        self.__codes = codes
        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__streams = src.functions.streams.Streams()

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
        messages = dask.compute(computations, scheduler='threads', num_workers=8)[0]

        return messages
