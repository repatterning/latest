"""Module data.py"""
import datetime
import logging

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
        self.__dtype = {'timestamp': np.float64, 'quality_code': np.float64, 'ts_id': np.float64, 'measure': float}

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

    @staticmethod
    def __set_missing(data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward filling.  In contrast, the variational model inherently deals with missing data, hence
                          it does not include this type of step.

        :param data:
        :return:
        """

        states = data['measure'].isna()
        logging.info(data.loc[states, :])

        data['measure'] = data['measure'].ffill().values

        return data


    def exc(self, listing: list[str]) -> pd.DataFrame:
        """

        :param listing:
        :return:
        """

        # The data
        data = self.__get_data(listing=listing)

        # Filter
        data: pd.DataFrame = data.copy().loc[data['timestamp'] >= self.__as_from, :]
        data.sort_values(by=['timestamp', 'quality_code'], ascending=True, inplace=True)
        data.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        data.drop(columns='quality_code', inplace=True)

        # Append a date of the format datetime64[]
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Missing data
        if sum(data['measure'].isna()) > 0:
            data = self.__set_missing(data=data.copy())

        return data
