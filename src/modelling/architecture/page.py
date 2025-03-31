"""Module page.py"""
import logging
import os

import statsmodels.tsa.forecasting.stl as tfs

import src.functions.objects


class Page:
    """
    <b>Notes</b><br>
    ------<br>

    This class saves the details of an institution's seasonal component model.
    """

    def __init__(self, system: tfs.STLForecastResults, path: str):
        """

        :param system: The forecasts/predictions of the seasonal component model.<br>
        :param path: The storage path.<br>
        """

        self.__system = system

        self.__path = path

    def __latex(self):
        """

        :return:
        """

        pathstr = os.path.join(self.__path, 'measures.tex')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__system.summary().as_latex())
            logging.info('success: %s', pathstr)
        except IOError as err:
            raise err from err

    def __txt(self):
        """

        :return:
        """

        pathstr = os.path.join(self.__path, 'measures.txt')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__system.summary().as_text())
            logging.info('success: %s', pathstr)
        except IOError as err:
            raise err from err

    def __extra(self):
        """

        :return:
        """

        nodes = {
            "parameters_estimation_method": getattr(self.__system, 'parameters_estimation_method'),
            "cov_type": self.__system.model_result.cov_type
        }

        # Path
        pathstr = os.path.join(self.__path, 'extra.json')

        # Persist
        message = src.functions.objects.Objects().write(nodes=nodes, path=pathstr)
        logging.info(message)

    def exc(self):
        """

        :return:
        """

        self.__latex()
        self.__txt()
        self.__extra()
