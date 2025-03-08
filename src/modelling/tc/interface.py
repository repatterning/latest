
import src.elements.codes as ce
import src.elements.master as mr

import src.modelling.tc.algorithm


class Interface:

    def __init__(self, arguments: dict):

        self.__arguments = arguments

    def exc(self, master: mr.Master, code: ce.Codes, state: bool) -> str:
        """

        :param master
        :param code:
        :param state:
        :return:
        """

        algorithm = src.modelling.tc.algorithm.Algorithm(training=master.training, arguments=self.__arguments)

        if state:
            return f'Trend Component Modelling: Proceed -> {code.hospital_code}'

        return f'Trend Component Modelling: Skip -> {code.hospital_code}'
