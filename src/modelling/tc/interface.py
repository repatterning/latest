import logging

import src.elements.codes as ce

class Interface:

    def __init__(self):
        pass

    @staticmethod
    def exc(code: ce.Codes, state: bool) -> str:
        """

        :param code:
        :param state:
        :return:
        """

        logging.info(state)

        if state:
            return f'Trend Component Modelling: Proceed -> {code.hospital_code}'

        return f'Trend Component Modelling: Skip -> {code.hospital_code}'
