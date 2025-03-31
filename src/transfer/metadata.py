"""Module metadata.py"""

import boto3

import src.functions.objects
import src.s3.configurations


class Metadata:
    """
    Notes<br>
    --------<br>

    This class reads-in the metadata of this project's data & references.<br><br>

    """

    def __init__(self, connector: boto3.session.Session):
        """

        :param connector: An instance of boto3.session.Session
        """

        self.__connector = connector

    def exc(self, name: str) -> dict:
        """

        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=self.__connector).objects(
            key_name=f'artefacts/metadata/{name}')

        return dictionary
