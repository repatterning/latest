"""Module interface.py"""
import logging
import os

import pandas as pd

import config
import src.elements.master as mr
import src.elements.partitions as pr
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

        self.__configurations = config.Config()

    def __frequency(self, training: pd.DataFrame):
        """

        :param training:
        :return:
        """


        training.set_index(keys='date', drop=False, inplace=True)
        training.index.rename('index', inplace=True)
        training.sort_index(axis=0, ascending=True, inplace=True)

        try:
            training.index.freq = self.__arguments.get('frequency')
        except ValueError:
            return pd.DataFrame()

        return training

    def exc(self, master: mr.Master, partition: pr.Partitions) -> str:
        """

        :param master: A named tuple consisting of a gauge's training & testing data.<br>
        :param partition: Encodes the time series & catchment identification codes of a gauge.<br>
        :return:
        """

        master.training.info()

        path = os.path.join(self.__configurations.assets_, str(partition.catchment_id), str(partition.ts_id))

        # Structuring
        _training =  self.__frequency(training=master.training.copy())
        if _training.empty:
            logging.info('Skipping %s of %s -> frequency issues.', partition.ts_id, partition.catchment_id)
            return f'Frequency issues: {partition.ts_id} of {partition.catchment_id} (no model)'

        # The model
        algorithm = src.modelling.architecture.algorithm.Algorithm(
            training=_training, arguments=self.__arguments, partition=partition)
        system = algorithm.exc()

        if system is None:
            return f'Unable to develop a model for {partition.ts_id} of {partition.catchment_id}'

        # Next, extract forecasts/predictions and supplementary details, subsequently persist; via the
        # model's <page> & <forecasts>.
        src.modelling.architecture.page.Page(system=system, path=path).exc()
        message = src.modelling.architecture.forecasts.Forecasts(
            master=master, arguments=self.__arguments, system=system, path=path).exc(partition=partition)

        return message
