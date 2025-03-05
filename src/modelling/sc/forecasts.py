"""Module forecasts.py"""
import logging
import os

import pandas as pd
import statsmodels.tsa.forecasting.stl as tfc

import config
import src.elements.codes as ce
import src.functions.objects


class Forecasts:

    def __init__(self, data: pd.DataFrame, testing: pd.DataFrame, system: tfc.STLForecastResults):
        """

        :param data:
        :param testing:
        :param system: The results of the seasonal component model
        """

        self.__data = data
        self.__testing = testing
        self.__system = system

        self.__configurations = config.Config()
        self.__objects = src.functions.objects.Objects()

    def __estimates(self) -> dict:
        """

        :return:
        """

        values: pd.DataFrame = self.__system.result.seasonal.to_frame()
        values.rename(columns={'season': 'seasonal_est'}, inplace=True)
        values = self.__data.copy()[['seasonal']].join(values.copy())
        values['date'] = values.index.strftime(date_format='%Y-%m-%d')

        values.reset_index(drop=True, inplace=True)

        return values.to_dict(orient='tight')

    def __tests(self, projections: pd.DataFrame) -> dict:
        """

        :param projections: Of test elements, for evaluations of errors
        :return:
        """

        values = self.__testing.copy()[['seasonal']].join(projections.copy())
        values['date'] = values.index.strftime(date_format='%Y-%m-%d')

        values.reset_index(drop=True, inplace=True)

        return values.to_dict(orient='tight')

    @staticmethod
    def __futures(projections: pd.DataFrame) -> dict:
        """

        :param projections: Of future elements of unknown value
        :return:
        """

        values = projections.copy()
        values['date'] = values.index.strftime(date_format='%Y-%m-%d')

        values.reset_index(drop=True, inplace=True)

        return values.to_dict(orient='tight')

    def exc(self, arguments: dict, code: ce.Codes):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        :param code: The identification code of a health board (board), and the identification code of an
                     institution/hospital (institution)
        :return:
        """

        steps = (2 * arguments.get('ahead'))
        forecasts = self.__system.forecast(steps=steps).to_frame()
        forecasts.rename(columns={0: 'seasonal_est'}, inplace=True)

        # Hence, the seasonal forecasts (sfc)
        nodes = {
            'health_board_code': code.health_board_code,
            'hospital_code': code.hospital_code,
            'estimates': self.__estimates(),
            'tests': self.__tests(projections=forecasts[-steps:-arguments.get('ahead')]),
            'futures': self.__futures(projections=forecasts[-arguments.get('ahead'):])
        }

        message = self.__objects.write(
            nodes=nodes,
            path=os.path.join(self.__configurations.artefacts_, 'models', code.hospital_code, 'sfc.json'))

        logging.info(message)
