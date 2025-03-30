"""Module interface.py"""
import logging
import os

import pandas as pd

import config
import src.elements.gauge as ge
import src.elements.master as mr
import src.modelling.architecture.algorithm
import src.modelling.architecture.forecasts
import src.modelling.architecture.page
import src.functions.directories


class Interface:
    """
    The seasonal component modelling interface
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()

    def __set_path(self, gauge: ge.Gauge):

        path = os.path.join(self.__configurations.artefacts_, 'models',
                            str(gauge.catchment_id), str(gauge.ts_id))
        self.__directories.create(path=path)

        return path

    def __restructure(self, training: pd.DataFrame):

        training.set_index(keys='date', inplace=True)
        training.sort_index(axis=0, ascending=True, inplace=True)
        training.index.freq = self.__arguments.get('frequency')

        return training

    def exc(self, master: mr.Master, gauge: ge.Gauge) -> str:
        """

        :param master: A named tuple consisting of a gauge's training & testing data.<br>
        :param gauge: Encodes the time series & catchment identification codes of a gauge.<br>
        :return:
        """

        path = self.__set_path(gauge=gauge)

        # The forecasting algorithm
        _training =  self.__restructure(training=master.training.copy())
        algorithm = src.modelling.architecture.algorithm.Algorithm(training=_training, arguments=self.__arguments, gauge=gauge)
        system = algorithm.exc()

        if system is None:
            return f'Unable to develop a model for {gauge.ts_id} of {gauge.catchment_id}'

        # Next, extract forecasts/predictions and supplementary details, subsequently persist; via the
        # model's <page> & <forecasts>.
        src.modelling.architecture.page.Page(system=system, path=path).exc()
        message = src.modelling.architecture.forecasts.Forecasts(
            master=master, arguments=self.__arguments, system=system, path=path).exc(gauge=gauge)

        return message
