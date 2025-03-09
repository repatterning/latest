"""Module interface.py"""
import logging

import src.elements.codes as ce
import src.elements.master as mr
import src.modelling.tc.algorithm
import src.modelling.tc.page


class Interface:

    def __init__(self, arguments: dict):

        self.__arguments = arguments

    def exc(self, master: mr.Master, code: ce.Codes, state: bool) -> str:
        """

        :param master: The training & testing data of an institution.
        :param code: The health board & institution/hospital codes of an institution/hospital.
        :param state: If the seasonal component modelling step was a success, this will be True; otherwise False.
        :return:
        """

        if not state:
            return f'Trend Component Modelling: Skip -> {code.hospital_code}'

        # Determine
        algorithm = src.modelling.tc.algorithm.Algorithm(training=master.training, arguments=self.__arguments)
        model, gp, details = algorithm.exc()

        # Persist: Model Algorithm
        src.modelling.tc.page.Page(model=model, code=code).exc()

        logging.info(type(gp))
        logging.info(type(details))


        return f'Trend Component Modelling: Success -> {code.hospital_code}'
