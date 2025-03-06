"""Module page.py"""
import logging
import os

import statsmodels.tsa.forecasting.stl as tfc

import config
import src.elements.codes as ce


class Page:
    """
    <b>Notes</b><br>
    ------<br>

    This class saves the details of an institution's seasonal component model.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

    def exc(self, system: tfc.STLForecastResults, code: ce.Codes):
        """

        :param system: The results of the seasonal component model
        :param code: The identification code of an institution
        :return:
        """

        pathstr = os.path.join(self.__configurations.artefacts_, 'models', code.hospital_code, 'scf.txt')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(system.summary().as_text())
            logging.info('scf.txt: succeeded (%s)', code.hospital_code)
        except IOError as err:
            raise err from err
