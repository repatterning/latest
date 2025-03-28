
import dask.dataframe as ddf
import pandas as pd


class Read:

    def __init__(self):
        pass

    @staticmethod
    def exc(sections: list) -> pd.DataFrame:

        try:
            data = ddf.read_csv(urlpath=sections)
        except ImportError as err:
            raise err from err

        return data.compute()
