"""Module interface.py"""
import logging
import os

import boto3
import pandas as pd

import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.ingress
import src.transfer.dictionary
import src.transfer.metadata


class Interface:
    """
    Notes<br>
    ------<br>

    This class executes the data transfer step; transfer to Amazon S3 (Simple Storage Service).
    """

    def __init__(self, connector: boto3.session.Session, service: sr.Service,  s3_parameters: s3p):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters

        # Metadata
        self.__metadata = src.transfer.metadata.Metadata(connector=connector)

        # Instances
        self.__dictionary = src.transfer.dictionary.Dictionary()

    def __get_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        """

        :param frame:
        :return:
        """

        _ingress: dict = self.__metadata.exc(name='ingress.json')
        _egress: dict = self.__metadata.exc(name='egress.json')

        frame = frame.assign(
            metadata = frame['section'].apply(
                lambda x: _ingress if x == 'data' else _egress))

        return frame

    def exc(self):
        """

        :return:
        """

        # The strings for transferring data to Amazon S3 (Simple Storage Service)
        strings = self.__dictionary.exc(
            path=os.path.join(os.getcwd(), 'warehouse'), extension='*', prefix='')

        # Adding metadata details per instance
        strings = self.__get_metadata(frame=strings.copy())
        logging.info(strings)

        # Transfer
        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.internal).exc(
            strings=strings, tagging='project=emergency')
        logging.info(messages)
