"""Module page.py"""
import logging
import os
import pathlib

import statsmodels.tsa.forecasting.stl as tfc

import config
import src.elements.codes as ce


class Page:
    """
    <b>Notes</b><br>
    ------<br>

    This class saves the details of an institution's seasonal component model.
    """

    def __init__(self, system: tfc.STLForecastResults, code: ce.Codes):
        """

        :param system: The results of the seasonal component model
        :param code: The identification code of an institution
        """

        self.__system = system
        self.__code = code

        configurations = config.Config()
        self.__root = os.path.join(configurations.artefacts_, 'models', code.hospital_code)

    def __latex(self):
        """

        :return:
        """

        pathstr = os.path.join(self.__root, 'scf_measures.tex')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__system.summary().as_latex())
            logging.info('%s: succeeded (%s)', pathlib.PurePath(pathstr).name, self.__code.hospital_code)
        except IOError as err:
            raise err from err

    def __txt(self):
        """

        :return:
        """

        pathstr = os.path.join(self.__root, 'scf_measures.txt')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__system.summary().as_text())
            logging.info('%s: succeeded (%s)', pathlib.PurePath(pathstr).name, self.__code.hospital_code)
        except IOError as err:
            raise err from err

    def exc(self):
        """

        :return:
        """

        self.__txt()
        self.__latex()
