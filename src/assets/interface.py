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

    @staticmethod
    def __date_formatting(blob: pd.DataFrame) -> pd.DataFrame:
        """

        :param blob:
        :return:
        """

        blob['week_ending_date'] = pd.to_datetime(
            blob['week_ending_date'].astype(dtype=str), errors='coerce', format='%Y-%m-%d')

        return blob

    def __get_uri(self, catchment_id, ts_id, datestr):
        """

        :param catchment_id:
        :param ts_id:
        :param datestr:
        :return:
        """

        return (f's3://{self.__s3_parameters.internal}/data/series/' + catchment_id.astype(str) +
         '/' + ts_id.astype(str) + '/' + datestr.astype(str) + '.csv')


    def exc(self):
        """
        url = f's3://{self.__s3_parameters.internal}/'

        :return:
        """

        gauges = src.assets.gauges.Gauges(service=self.__service, s3_parameters=self.__s3_parameters).exc()

        values: pd.DataFrame = gauges[['catchment_id']].groupby(by='catchment_id').value_counts().to_frame()
        values = values.copy().loc[values['count'].isin(self.__arguments.get('catchments').get('chunks')), :]
        values.reset_index(drop=False, inplace=True)

        gauges = gauges.copy().loc[gauges['catchment_id'].isin(values['catchment_id'].values), :]
        logging.info(gauges)

        partitions = src.assets.partitions.Partitions(data=gauges).exc(arguments=self.__arguments)
        partitions['uri'] = self.__get_uri(partitions['catchment_id'], partitions['ts_id'], partitions['datestr'])
        logging.info(partitions)
