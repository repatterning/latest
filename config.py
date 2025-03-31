"""
Module config.py
"""
import datetime
import logging
import os


class Config:
    """
    Description
    -----------

    A class for configurations
    """

    def __init__(self) -> None:
        """
        <b>Notes</b><br>
        ------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, *key, etc.<br><br>

        """

        '''
        Date Stamp
        '''
        now = datetime.datetime.now()
        self.stamp: str = now.strftime('%Y-%m-%d')
        logging.info(self.stamp)


        '''
        The key of the Amazon S3 (Simple Storage Service) parameters.
        '''
        self.s3_parameters_key = 's3_parameters.yaml'


        '''
        Local Paths
        '''
        self.warehouse: str = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_: str = os.path.join(self.warehouse, 'artefacts', self.stamp)
