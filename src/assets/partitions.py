"""Module partitions.py"""
import datetime
import logging
import sys
import typing

import pandas as pd

import src.functions.cache


class Partitions:
    """
    Partitions for parallel computation.
    """

    def __init__(self, gauges: pd.DataFrame, foci: pd.DataFrame, arguments: dict):
        """

        :param gauges:
        :param foci:
        :param arguments:
        """

        self.__gauges = gauges
        self.__foci = foci
        self.__arguments = arguments

    def __limits(self):
        """

        :return:
        """

        # The boundaries of the dates; datetime format
        spanning = self.__arguments.get('spanning')
        as_from = datetime.date.today() - datetime.timedelta(days=round(spanning*365))
        starting = datetime.datetime.strptime(f'{as_from.year}-01-01', '%Y-%m-%d')

        _end = datetime.datetime.now().year
        ending = datetime.datetime.strptime(f'{_end}-01-01', '%Y-%m-%d')

        # Create series
        limits = pd.date_range(start=starting, end=ending, freq='YS'
                              ).to_frame(index=False, name='date')

        return limits

    def exc(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """

        :return:
        """

        # The years in focus, via the year start date, e.g., 2023-01-01
        limits = self.__limits()

        # Inspecting ...
        codes = self.__gauges.merge(self.__foci[['catchment_id', 'ts_id']], how='right', on=['catchment_id', 'ts_id'])
        if codes.shape[0] == 0:
            logging.info('None valid codes.')
            src.functions.cache.Cache().exc()
            sys.exit(0)

        # Hence, the gauges in focus vis-à-vis the years in focus
        listings = limits.merge(codes, how='left', on='date')
        partitions = listings[['catchment_id', 'ts_id']].drop_duplicates()

        return partitions, listings
