"""Module s3_parameters.py"""

import boto3
import yaml

import config
import src.elements.s3_parameters as s3p
import src.functions.secret
import src.s3.unload


class S3Parameters:
    """
    Notes<br>
    --------<br>

    This class reads-in the YAML file of this project repository's overarching Amazon S3 (Simple Storage Service)
    parameters.<br><br>

    <a href="https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-express-Regions-and-Zones.html" target="_blank">
    S3 Express One Zone, which has 4 overarching regions.</a>

    """

    def __init__(self, connector: boto3.session.Session):
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        """

        # An instance for S3 interactions
        self.__s3_client: boto3.session.Session.client = connector.client(
            service_name='s3')

        # Hence
        self.__configurations = config.Config()
        self.__secret = src.functions.secret.Secret(connector=connector)

    def __get_dictionary(self) -> dict:
        """

        :return:
            A dictionary, or excerpt dictionary, of YAML file contents
        """

        buffer = src.s3.unload.Unload(s3_client=self.__s3_client).exc(
            bucket_name=self.__secret.exc(secret_id='AccidentEmergency', node='configurations'),
            key_name=self.__configurations.s3_parameters_key)

        try:
            data: dict = yaml.load(stream=buffer, Loader=yaml.CLoader)
        except yaml.YAMLError as err:
            raise err from err

        return data['parameters']

    def __build_collection(self, dictionary: dict) -> s3p.S3Parameters:
        """

        :param dictionary:
        :return:
            A re-structured form of the parameters.
        """

        s3_parameters = s3p.S3Parameters(**dictionary)

        # Parsing variables
        region_name = self.__secret.exc(secret_id='RegionCodeDefault')
        internal = self.__secret.exc(secret_id='AccidentEmergency', node='internal')
        configurations = self.__secret.exc(secret_id='AccidentEmergency', node='configurations')

        s3_parameters: s3p.S3Parameters = s3_parameters._replace(
            location_constraint=region_name, region_name=region_name, internal=internal, configurations=configurations)

        return s3_parameters

    def exc(self) -> s3p.S3Parameters:
        """

        :return:
            The re-structured form of the parameters.
        """

        dictionary = self.__get_dictionary()

        return self.__build_collection(dictionary=dictionary)
