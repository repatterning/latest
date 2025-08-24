"""Module data.py"""
import datetime

import dask.dataframe as ddf
import numpy as np
import pandas as pd


class Data:
    """
    Data
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        # Focus
        self.__dtype = {'timestamp': np.float64, 'ts_id': np.float64, 'measure': float}

        # seconds, milliseconds
        as_from: datetime.datetime = (datetime.datetime.now()
                                      - datetime.timedelta(days=round(arguments.get('spanning')*365)))
        self.__as_from = as_from.timestamp() * 1000

    def __get_data(self, listing: list[str]):
        """

        :param listing:
        :return:
        """

        try:
            block: pd.DataFrame = ddf.read_csv(
                listing, header=0, usecols=list(self.__dtype.keys()), dtype=self.__dtype).compute()
        except ImportError as err:
            raise err from err

        block.reset_index(drop=True, inplace=True)

        return block

    def exc(self, listing: list[str]) -> pd.DataFrame:
        """

        :param listing:
        :return:
        """

        # The data
        data = self.__get_data(listing=listing)

        # Filter
        data = data.copy().loc[data['timestamp'] >= self.__as_from, :]

        # Append a date of the format datetime64[]
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

        return data
