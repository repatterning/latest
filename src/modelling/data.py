import datetime
import time


import dask.dataframe as ddf
import pandas as pd


class Data:

    def __init__(self, arguments: dict):

        spanning = arguments.get('spanning')
        as_from = datetime.date.today() - datetime.timedelta(days=round(spanning*365))
        self.__starting = 1000 * time.mktime(as_from.timetuple())

    def __setting_up(self, frame: pd.DataFrame) -> pd.DataFrame:

        frame['date'] = pd.to_datetime(frame['timestamp'], unit='ms')

        return frame.loc[frame['timestamp'] >= self.__starting, :]

    def exc(self, sections: list) -> pd.DataFrame:

        try:
            data = ddf.read_csv(urlpath=sections)
        except ImportError as err:
            raise err from err

        frame: pd.DataFrame = data.compute()
        frame = self.__setting_up(frame=frame.copy())

        return frame
