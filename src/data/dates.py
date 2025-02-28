"""Module dates.py"""
import logging
import typing

import dask
import pandas as pd


class Dates:
    """
    Notes<br>
    ------<br>

    Unfortunately, the delivery day of the week is uncertain.  Hence, this project will initially
    assume that the points of an institution's series are contiguous.  Do not use this class yet.
    """

    def __init__(self, data: pd.DataFrame):
        """

        :param data:
        """

        self.__frame = data.copy()

    def __limits(self) -> typing.Tuple[pd.Timestamp, pd.Timestamp]:
        """

        :return:
        """

        minimum = self.__frame['week_ending_date'].min()
        maximum = self.__frame['week_ending_date'].max()

        return minimum, maximum

    @staticmethod
    def __indices(minimum: pd.Timestamp, maximum: pd.Timestamp) -> pd.DataFrame:
        """

        :param minimum: Start Date
        :param maximum: End Date
        :return:
        """

        indices: pd.DataFrame = pd.date_range(start=minimum, end=maximum, inclusive='both', freq='W').to_frame()
        indices.reset_index(drop=True, inplace=True)
        indices.rename(columns={0: 'week_ending_date'}, inplace=True)
        logging.info(indices)

        return indices

    @dask.delayed
    def __dates(self, indices: pd.DataFrame, code: str) -> pd.DataFrame:
        """

        :param indices: Dates
        :param code: Hospital Code
        :return:
        """

        blob = self.__frame.loc[self.__frame['hospital_code'] == code, :]

        # Common Values
        fields = ['health_board_code', 'hospital_code', 'department_type', 'attendance_category']
        reference = blob[fields].drop_duplicates().values

        # Ascertaining all date points within a range
        frame = indices.merge(blob, how='left', on='week_ending_date')
        frame.loc[:, fields] = reference

        return frame

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        minimum, maximum = self.__limits()
        indices = self.__indices(minimum=minimum, maximum=maximum)
        codes = self.__frame['hospital_code'].unique()

        computations = []
        for code in codes:
            frame = self.__dates(indices=indices, code=code)
            computations.append(frame)
        calculations = dask.compute(computations, scheduler='threads')[0]
        data = pd.concat(calculations, axis=0, ignore_index=True)
        logging.info(data)

        return data
