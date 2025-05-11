"""Module forecasts.py"""
import datetime
import os

import numpy as np
import pandas as pd
import statsmodels.tsa.forecasting.stl as tfs

import src.elements.partitions as pr
import src.elements.master as mr
import src.functions.objects
import src.modelling.architecture.restructure


class Forecasts:
    """
    Determines forecasts/predictions vis-à-vis a developed model; predictions.
    """

    def __init__(self, master: mr.Master, arguments: dict, system: tfs.STLForecastResults, path: str):
        """

        :param master: A named tuple consisting of training & testing data.<br>
        :param arguments: A set of model development, and supplementary, arguments.<br>
        :param system: The forecasting object.<br>
        :param path: The storage path.<br>
        """

        self.__master = master
        self.__arguments = arguments
        self.__system = system
        self.__path = path

        self.__objects = src.functions.objects.Objects()
        self.__restructure = src.modelling.architecture.restructure.Restructure()

    def __get_predictions(self, limit: datetime.datetime) -> pd.DataFrame:
        """

        :param limit: Predictions will be until limit.
        :return:
        """

        details = self.__system.get_prediction(end=limit)

        # Final State -> ['date', 'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']
        predictions = details.summary_frame()
        predictions.reset_index(drop=False, inplace=True)
        predictions.rename(columns={'index': 'date'}, inplace=True)

        return predictions

    def __get_training(self, predictions: pd.DataFrame) -> dict:
        """

        :param predictions:
        :return:
        """

        _training = self.__master.training[['timestamp', 'date', 'measure']].merge(predictions, how='left', on='date')
        _training.drop(columns='date', inplace=True)

        return self.__restructure.exc(data=_training.copy())

    def __get_testing(self, predictions: pd.DataFrame) -> dict:
        """

        :param predictions:
        :return:
        """

        _testing = self.__master.testing[['timestamp', 'date', 'measure']].merge(predictions, how='left', on='date')
        _testing.drop(columns='date', inplace=True)

        return self.__restructure.exc(data=_testing.copy())

    def __get_futures(self, predictions: pd.DataFrame) -> dict:
        """

        :param predictions:
        :return:
        """

        _futures: pd.DataFrame = predictions.copy()[-self.__arguments.get('ahead'):]
        _futures['timestamp'] = _futures['date'].astype(np.int64)//(10**6)
        _futures.drop(columns='date', inplace=True)

        return self.__restructure.exc(data=_futures.copy())

    def exc(self,  partition: pr.Partitions) -> str:
        """

        :param partition: Encodes the time series & catchment identification codes of a gauge, and its gauge datum.<br>
        :return:
        """

        hours = self.__arguments.get('testing') + self.__arguments.get('ahead')
        limit: datetime.datetime = (self.__master.training['date'].max().to_pydatetime() +
                 datetime.timedelta(hours=hours))
        predictions = self.__get_predictions(limit=limit)

        # Hence
        nodes = {'ts_id': partition.ts_id,
                 'catchment_id': partition.catchment_id,
                 'training': self.__get_training(predictions=predictions),
                 'testing': self.__get_testing(predictions=predictions),
                 'futures': self.__get_futures(predictions=predictions)}
        message = self.__objects.write(nodes=nodes, path=os.path.join(self.__path, 'estimates.json'))

        return f'{message} ({partition.ts_id} of {partition.catchment_id})'
