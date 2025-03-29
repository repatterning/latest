"""Module forecasts.py"""
import logging
import os
import datetime

import pandas as pd
import statsmodels.tsa.forecasting.stl as tfs

import config
import src.elements.gauge as ge
import src.elements.master as mr
import src.functions.objects


class Forecasts:
    """
    Determines forecasts/predictions vis-à-vis a developed model; predictions.
    """

    def __init__(self, master: mr.Master, arguments: dict, system: tfs.STLForecastResults):
        """

        :param master: A named tuple consisting of an institutions training & testing data
        :param arguments: A set of model development, and supplementary, arguments.
        :param system: The results of the seasonal component model
        """

        self.__training = master.training
        self.__testing = master.testing
        self.__arguments = arguments
        self.__system = system

        self.__configurations = config.Config()
        self.__objects = src.functions.objects.Objects()

    def __get_estimations(self) -> pd.DataFrame:

        limit = self.__training['date'].max().to_pydatetime() + datetime.timedelta(hours=2*self.__arguments.get('ahead'))
        details = self.__system.get_prediction(end=limit)

        # Final State -> ['date', 'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']
        estimations = details.summary_frame()
        estimations.reset_index(drop=False, inplace=True)
        estimations.rename(columns={'index': 'date'}, inplace=True)

        return estimations

    def __get_training(self, estimations: pd.DataFrame):

        _training = self.__training[['ts_id', 'timestamp', 'date', 'measure']].merge(estimations, how='left', on='date')
        return _training.to_dict(orient='tight')

    def __get_testing(self, estimations: pd.DataFrame):

        _testing = self.__testing[['ts_id', 'timestamp', 'date', 'measure']].merge(estimations, how='left', on='date')
        _testing.to_dict(orient='tight')

    def __get_futures(self, estimations: pd.DataFrame):

        _futures: pd.DataFrame = estimations[-self.__arguments.get('ahead'):]
        _futures.to_dict(orient='tight')


    def exc(self,  gauge: ge.Gauge) -> str:
        """

        :param gauge: Encodes the time series & catchment identification codes of a gauge, and its gauge datum.<br>
        :return:
        """

        estimations = self.__get_estimations()

        # Hence
        nodes = {
            'ts_id': gauge.ts_id,
            'catchment_id': gauge.catchment_id,
            'training': self.__get_training(estimations=estimations),
            'testing': self.__get_testing(estimations=estimations),
            'futures': self.__get_futures(estimations=estimations)}

        root = os.path.join(
            self.__configurations.artefacts_, 'models', str(gauge.catchment_id), str(gauge.ts_id))
        message = self.__objects.write(nodes=nodes, path=os.path.join(root, 'scf_estimates.json'))

        return f'{message} ({gauge.ts_id} of {gauge.catchment_id})'
