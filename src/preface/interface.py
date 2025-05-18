"""Module interface.py"""
import typing

import boto3

import config
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.service
import src.preface.setup
import src.s3.configurations
import src.s3.s3_parameters


class Interface:
    """
    Interface

    https://docs.python.org/3/library/multiprocessing.html
    https://superfastpython.com/multiprocessing-in-python
    https://superfastpython.com/multiprocessing-pool-python/
    """

    def __init__(self):
        """

        """

        self.__configurations = config.Config()

    def __get_arguments(self, connector: boto3.session.Session) -> dict:
        """

        :return:
        """

        key_name = self.__configurations.arguments_key

        arguments = src.s3.configurations.Configurations(connector=connector).objects(key_name=key_name)

        return arguments

    def exc(self, codes: list[int] | None) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, dict]:
        """

        :param codes:
        :return:
        """

        connector = boto3.session.Session()
        s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
        service: sr.Service = src.functions.service.Service(
            connector=connector, region_name=s3_parameters.region_name).exc()
        arguments: dict = self.__get_arguments(connector=connector)

        if codes is not None:
            arguments['series']['excerpt'] = codes

        src.preface.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

        return connector, s3_parameters, service, arguments
