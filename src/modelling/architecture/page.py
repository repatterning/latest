"""Module page.py"""
import logging
import os
import pathlib

import statsmodels.tsa.forecasting.stl as tfs

import config
import src.elements.gauge as ge
import src.functions.objects


class Page:
    """
    <b>Notes</b><br>
    ------<br>

    This class saves the details of an institution's seasonal component model.
    """

    def __init__(self, system: tfs.STLForecastResults, gauge: ge.Gauge):
        """

        :param system: The forecasts/predictions of the seasonal component model.
        :param gauge: Encodes the time series & catchment identification codes of a gauge, and its gauge datum.<br>
        """

        self.__system = system
        self.__gauge = gauge

        configurations = config.Config()
        self.__root = os.path.join(configurations.artefacts_, 'models',
                                   str(self.__gauge.catchment_id), str(self.__gauge.ts_id))

    def __latex(self):
        """

        :return:
        """

        pathstr = os.path.join(self.__root, 'scf_measures.tex')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__system.summary().as_latex())
            logging.info('%s: succeeded (%s of %s)', pathlib.PurePath(pathstr).name,
                         self.__gauge.catchment_id, self.__gauge.ts_id)
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
            logging.info('%s: succeeded (%s of %s)', pathlib.PurePath(pathstr).name,
                         self.__gauge.catchment_id, self.__gauge.ts_id)
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
        pathstr = os.path.join(self.__root, 'scf_extra.json')

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
