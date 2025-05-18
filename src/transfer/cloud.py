"""Module setup.py"""
import sys

import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.cache
import src.functions.directories
import src.s3.bucket
import src.s3.keys
import src.s3.prefix


class Cloud:
    """
    Description
    -----------

    Sets up local & cloud environments
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this project, e.g., region code
                              name, buckets, etc.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__bucket_name = self.__s3_parameters.internal

    def __s3(self) -> bool:
        """
        Prepares an Amazon S3 (Simple Storage Service) bucket.

        :return:
        """

        # An instance for interacting with Amazon S3 buckets.
        bucket = src.s3.bucket.Bucket(service=self.__service, location_constraint=self.__s3_parameters.location_constraint,
                                      bucket_name=self.__bucket_name)

        # Strategy Switch: If the bucket exist, do not clear the target prefix, overwrite files instead.
        if bucket.exists():
            return True

        return bucket.create()

    def exc(self) -> bool:
        """

        :return:
        """

        if self.__s3():
            return True

        src.functions.cache.Cache().exc()
        sys.exit('Unable to set up an Amazon S3 (Simple Storage Service) section.')
