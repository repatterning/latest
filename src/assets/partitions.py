"""Module partitions.py"""
import typing
import logging
import datetime
import numpy as np
import pandas as pd


class Partitions:
    """
    Partitions for parallel computation.
    """

    def __init__(self, data: pd.DataFrame, arguments: dict):
        """

        :param data:
        :param arguments:
        """

        self.__data = data
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
        logging.info(limits)

        # Focusing on ...
        excerpt = self.__arguments.get('series').get('excerpt')
        if excerpt is None:
            data =  self.__data
        else:
            codes = np.unique(np.array(excerpt))
            data = self.__data.copy().loc[self.__data['ts_id'].isin(codes), :]
            data = data if data.shape[0] > 0 else self.__data

        # Hence, the data sets in focus vis-à-vis the years in focus
        listings = limits.merge(data, how='left', on='date')

        # ...
        partitions = listings[['catchment_id', 'ts_id']].drop_duplicates()

        return partitions, listings
