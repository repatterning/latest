"""Module interface.py"""
import logging

import pandas as pd

import src.elements.gauge as ge
import src.elements.master as mr
import src.modelling.architecture.algorithm
import src.modelling.architecture.forecasts
import src.modelling.architecture.page


class Interface:
    """
    The seasonal component modelling interface
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__arguments = arguments

    def exc(self, master: mr.Master, gauge: ge.Gauge) -> str:
        """

        :param master: A named tuple consisting of a gauge's training & testing data.<br>
        :param gauge: Encodes the time series & catchment identification codes of a gauge.<br>
        :return:
        """

        _training: pd.DataFrame = master.training.copy()
        _training.set_index(keys='date', inplace=True)
        _training.sort_index(axis=0, ascending=True, inplace=True)
        _training.index.freq = self.__arguments.get('frequency')

        # The forecasting algorithm
        algorithm = src.modelling.architecture.algorithm.Algorithm(training=_training, arguments=self.__arguments, gauge=gauge)
        system = algorithm.exc()

        if system is None:
            return f'Unable to develop a model for {gauge.ts_id} of {gauge.catchment_id}'

        # Next, extract forecasts/predictions and supplementary details, subsequently persist; via the
        # model's <page> & <forecasts>.
        src.modelling.architecture.page.Page(system=system, gauge=gauge).exc()
        message = src.modelling.architecture.forecasts.Forecasts(
            master=master, arguments=self.__arguments, system=system).exc(gauge=gauge)

        return message
