"""Module page.py"""
import os

import statsmodels.tsa.forecasting.stl as tfc

import config


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

    def exc(self, system: tfc.STLForecastResults, code: str):
        """

        :param system: The results of the seasonal component model
        :param code: The identification code of an institution
        :return:
        """

        pathstr = os.path.join(self.__configurations.artefacts_, 'models', code, 'sc.txt')

        with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
            disk.write(system.summary().as_text())
