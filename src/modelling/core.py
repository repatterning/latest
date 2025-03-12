
import glob
import os

import dask

import config
import src.elements.codes as ce
import src.elements.master as mr
import src.functions.streams
import src.modelling.tc.interface


class Core:

    def __init__(self, codes: list[ce.Codes], arguments: dict):
        """

        :param codes:
        :param arguments:
        """

        self.__codes = codes
        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__streams = src.functions.streams.Streams()

    def __get_codes(self) -> list[ce.Codes]:
        """

        :return:
        """

        strings = glob.glob(
            pathname=os.path.join(self.__configurations.artefacts_, 'data', '**', '*training.csv'))
        values = [os.path.basename(os.path.dirname(string)) for string in strings]

        return [code for code in self.__codes if code.hospital_code in values]

    def exc(self, masters: list[mr.Master]) -> list[str]:
        """

        :return:
        """

        tc = dask.delayed(src.modelling.tc.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for master in masters:
            message = tc(training=master.training)
            computations.append(message)
        messages = dask.compute(computations, scheduler='threads', num_workers=8)[0]

        return messages
