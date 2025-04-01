"""Module data.py"""
import logging
import datetime
import time

import dask.dataframe as ddf
import pandas as pd

import src.elements.gauge as ge


class Data:
    """
    A gauge's data
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        spanning = arguments.get('spanning')
        as_from = datetime.date.today() - datetime.timedelta(days=round(spanning*365))
        self.__starting = 1000 * time.mktime(as_from.timetuple())

    @staticmethod
    def __anomalies(frame: pd.DataFrame) -> pd.DataFrame:

        data = frame.copy().drop(columns='quality_code')
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        data.drop_duplicates(keep='last', inplace=True)

        return data

    def __setting_up(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        frame['date'] = pd.to_datetime(frame['timestamp'], unit='ms')

        return frame.loc[frame['timestamp'] >= self.__starting, :]

    @staticmethod
    def __timings(frame: pd.DataFrame):
        """

        :param frame:
        :return:
        """

        dates = pd.date_range(start=frame['date'].min(), end=frame['date'].max(),
                              freq='h', inclusive='left')
        slices = pd.DataFrame(data={'date': dates})

        data = slices.merge(frame, how='left', on='date')
        data.sort_values(by=['ts_id', 'timestamp'], ascending=True, inplace=True)

        return data

    def exc(self, sections: list, gauge: ge.Gauge) -> pd.DataFrame:
        """

        :param sections:
        :param gauge:
        :return:
        """

        logging.info(sections)

        try:
            data = ddf.read_csv(sections)
        except ImportError as err:
            raise err from err

        frame: pd.DataFrame = data.compute()

        # Anomalies
        frame = self.__anomalies(frame=frame)

        # Measure
        frame['measure'] = frame['value'] + gauge.gauge_datum

        # Setting up, focusing on hour points
        frame = self.__setting_up(frame=frame.copy())
        frame = self.__timings(frame=frame.copy())

        return frame
