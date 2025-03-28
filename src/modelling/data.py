
import dask.dataframe as ddf
import pandas as pd


class Data:

    def __init__(self):
        pass

    @staticmethod
    def __set_date(frame: pd.DataFrame):

        frame['date'] = pd.to_datetime(frame['timestamp'], unit='ms')

        return frame

    def exc(self, sections: list) -> pd.DataFrame:

        try:
            data = ddf.read_csv(urlpath=sections)
        except ImportError as err:
            raise err from err

        frame: pd.DataFrame = data.compute()

        return self.__set_date(frame=frame.copy())
