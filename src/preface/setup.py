"""Module setup.py"""
import sys

import config
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.cache
import src.functions.directories
import src.s3.bucket
import src.s3.keys
import src.s3.prefix


class Setup:
    """
    Description
    -----------

    Sets up local & cloud environments.  <b>If</b> the setting up of the local & cloud environments fail, the
    package exits here.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this project, e.g., region code
                              name, buckets, etc.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters

        # Configurations, etc.
        self.__configurations = config.Config()
        self.__prefix = s3_parameters.path_internal_artefacts + self.__configurations.stamp

        # Instances
        self.__directories = src.functions.directories.Directories()

    def __clear_prefix(self) -> bool:
        """

        :return:
        """

        # An instance for interacting with objects within an Amazon S3 prefix
        instance = src.s3.prefix.Prefix(service=self.__service, bucket_name=self.__s3_parameters.internal)

        # Get the keys therein
        keys: list[str] = instance.objects(prefix=self.__prefix)

        if len(keys) > 0:
            objects = [{'Key' : key} for key in keys]
            state = instance.delete(objects=objects)
            return bool(state)

        return True

    def __s3(self) -> bool:
        """
        Prepares an Amazon S3 (Simple Storage Service) bucket.

        :return:
        """

        # An instance for interacting with Amazon S3 buckets.
        bucket = src.s3.bucket.Bucket(service=self.__service, location_constraint=self.__s3_parameters.location_constraint,
                                      bucket_name=self.__s3_parameters.internal)

        # If the bucket exist, the prefix path is cleared.  Otherwise, the bucket is created.
        if bucket.exists():
            self.__clear_prefix()

        return bucket.create()

    def __local(self) -> bool:
        """

        :return:
        """

        self.__directories.cleanup(path=self.__configurations.warehouse)

        return self.__directories.create(path=self.__configurations.warehouse)

    def exc(self) -> bool:
        """

        :return:
        """

        if self.__s3() & self.__local():
            return True

        src.functions.cache.Cache().exc()
        sys.exit('No Executions')
