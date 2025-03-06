"""Module interface.py"""
import logging

import pandas as pd

import config
import src.elements.s3_parameters as s3p
import src.elements.text_attributes as txa
import src.functions.streams


class Interface:
    """
    Notes<br>
    ------<br>

    Reads-in the data in focus.
    """

    def __init__(self, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param s3_parameters: The overarching S3 parameters settings of this project, e.g., region code
                              name, buckets, etc.
        :param arguments:
        """

        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

        # An instance for writing/reading CSV (comma-separated values) files
        self.__streams = src.functions.streams.Streams()

    def __get_data(self) -> pd.DataFrame:
        """

        :return:
        """

        uri = ('s3://' + self.__s3_parameters.internal + '/' + self.__s3_parameters.path_internal_data +
               self.__configurations.source)
        text = txa.TextAttributes(uri=uri, header=0)
        data = self.__streams.read(text=text)

        return data[self.__configurations.fields]

    @staticmethod
    def __date_formatting(blob: pd.DataFrame) -> pd.DataFrame:
        """

        :param blob:
        :return:
        """

        blob['week_ending_date'] = pd.to_datetime(
            blob['week_ending_date'].astype(dtype=str), errors='coerce', format='%Y-%m-%d')

        return blob

    def __skip(self, b: pd.DataFrame):

        frame = b.copy()

        # Counting n_attendances values <= 0 per institution
        cases = frame[['hospital_code', 'n_attendances']].groupby('hospital_code').agg(
            missing=('n_attendances', lambda x: sum(x <= 0)))
        cases.reset_index(drop=False, inplace=True)
        cases: pd.DataFrame = cases.copy().loc[cases['missing'] > 0, :]

        # Skip institutions that have zero or negative values
        if not cases.empty:
            frame = frame.copy().loc[~frame['hospital_code'].isin(cases['hospital_code'].unique()), :]

        return frame

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        # The data
        data = self.__get_data()

        # Format dates
        data = self.__date_formatting(blob=data.copy())


        logging.info(data)

        return data
