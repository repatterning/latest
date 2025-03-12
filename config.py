"""
Module config.py
"""
import logging
import datetime
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
        Date Stamp: The most recent Tuesday.  The code of Tuesday is 1, hence 
        now.weekday() - 1
        '''
        now = datetime.datetime.now()
        offset = (now.weekday() - 1) % 7
        tuesday = now - datetime.timedelta(days=offset)
        self.stamp: str = tuesday.strftime('%Y-%m-%d')
        logging.info(self.stamp)


        '''
        The prefix.ending.string & key.name of the modelling data; ref.
            s3:// {bucket} / {prefix.starting.string} / {prefix.ending.string} / {key.name}
        Additionally, the path to the artefacts metadata.
        '''
        self.source = f'modelling/{self.stamp}.csv'
        self.artefacts_metadata_ = 'artefacts/metadata'


        '''
        The key of the Amazon S3 (Simple Storage Service) parameters.
        '''
        self.s3_parameters_key = 's3_parameters.yaml'


        '''
        Local Paths
        '''
        self.warehouse: str = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_: str = os.path.join(self.warehouse, 'artefacts', self.stamp)


        '''
        Extra
        The <boundary> is a possible data start-point cut-off, boundary.
        '''
        self.fields = ['week_ending_date', 'health_board_code', 'hospital_code', 'n_attendances']
        self.boundary = '2020-06-01'
