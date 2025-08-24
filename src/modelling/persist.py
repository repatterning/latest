"""Module persist.py"""
import os

import pandas as pd

import config
import src.elements.partitions as pr
import src.elements.inference as ifr
import src.functions.streams


class Persist:
    """
    Constructor
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()
        self.__streams = src.functions.streams.Streams()

    def __persist(self, blob: pd.DataFrame, path: str) -> str:
        """

        :param blob:
        :param path:
        :return:
        """

        return self.__streams.write(blob=blob, path=path)

    def exc(self, inference: ifr.Inference, partition: pr.Partitions) -> str:
        """

        :param inference:
        :param partition:
        :return:
        """


        endpoint = os.path.join(
            self.__configurations.assets_, str(partition.catchment_id), str(partition.ts_id))

        message = '|'.join((
            str(partition.catchment_id),
            str(partition.ts_id),
            self.__persist(blob=inference.evidence_lower_bound, path=os.path.join(endpoint, 'evidence_lower_bound.csv')),
            self.__persist(blob=inference.estimates, path=os.path.join(endpoint, 'estimates.csv'))
        ))

        return message
