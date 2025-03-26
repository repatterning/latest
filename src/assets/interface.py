"""Module interface.py"""
import logging

import pandas as pd

import config
import src.assets.gauges
import src.assets.partitions
import src.elements.s3_parameters as s3p
import src.elements.service as sr


class Interface:
    """
    Notes<br>
    ------<br>

    Reads-in the data in focus.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service:
        :param s3_parameters: The overarching S3 parameters settings of this project, e.g., region code
                              name, buckets, etc.
        :param arguments:
        """

        self.__service = service
        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

    def __get_uri(self, catchment_id, ts_id, datestr):
        """

        :param catchment_id:
        :param ts_id:
        :param datestr:
        :return:
        """

        return (f's3://{self.__s3_parameters.internal}/data/series/' + catchment_id.astype(str) +
         '/' + ts_id.astype(str) + '/' + datestr.astype(str) + '.csv')

    def __filter(self, gauges: pd.DataFrame) -> pd.DataFrame:

        values: pd.DataFrame = gauges[['catchment_id']].groupby(by='catchment_id').value_counts().to_frame()
        values = values.copy().loc[values['count'].isin(self.__arguments.get('catchments').get('chunks')), :]
        values.reset_index(drop=False, inplace=True)

        selection = gauges.copy().loc[gauges['catchment_id'].isin(values['catchment_id'].values), :]

        return selection

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        # Applicable time series, i.e., gauge, identification codes
        gauges = src.assets.gauges.Gauges(service=self.__service, s3_parameters=self.__s3_parameters).exc()
        if self.__arguments.get('catchments').get('chunks') is not None:
            gauges = self.__filter(gauges=gauges.copy())

        # Strings for data reading
        partitions: pd.DataFrame = src.assets.partitions.Partitions(data=gauges).exc(arguments=self.__arguments)
        partitions['uri'] = self.__get_uri(partitions['catchment_id'], partitions['ts_id'], partitions['datestr'])

        return partitions
