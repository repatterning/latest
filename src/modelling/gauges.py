"""Module codes.py"""
import pandas as pd

import src.elements.gauge as ge


class Gauges:
    """
    Determines the unique set of health board & institution pairings
    """

    def __init__(self):
        pass

    @staticmethod
    def __structure(values: list[dict]) -> list[ge.Gauge]:
        """

        :param values:
        :return:
        """

        return [ge.Gauge(**value) for value in values]

    def exc(self, assets: pd.DataFrame) -> list[ge.Gauge]:
        """

        :param assets:
        :return:
        """

        # Codes
        frame = assets[['catchment_id', 'ts_id']].drop_duplicates()
        values: list[dict] = frame.reset_index(drop=True).to_dict(orient='records')

        return self.__structure(values=values)
