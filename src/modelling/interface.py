"""Module interface.py"""
import logging
import os

import dask
import pandas as pd

import config
import src.elements.codes as ce
import src.elements.master as mr
import src.functions.directories
import src.modelling.codes
import src.modelling.decompose
import src.modelling.sc.interface
import src.modelling.splits
import src.modelling.core


class Interface:
    """
    Interface
    """

    def __init__(self, data: pd.DataFrame, arguments: dict):
        """

        :param data:
        :param arguments:
        """

        self.__data = data
        self.__arguments = arguments

        # Settings
        self.__codes: list[ce.Codes] = src.modelling.codes.Codes().exc(data=self.__data)

        # Instances
        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()

    @dask.delayed
    def __get_data(self, code: ce.Codes) -> pd.DataFrame:
        """

        :param code: The health board & institution/hospital codes of an institution/hospital.
        :return:
        """

        frame = self.__data.copy().loc[self.__data['hospital_code'] == code.hospital_code, :]
        frame.set_index(keys='week_ending_date', drop=True, inplace=True)
        frame.sort_values(by=['week_ending_date'], ascending=True, ignore_index=False, inplace=True)

        return frame

    @dask.delayed
    def __set_directories(self, code: ce.Codes) -> bool:
        """

        :param code: The health board & institution/hospital codes of an institution/hospital.
        :return:
        """

        success = []
        for pathway in ['data', 'models']:
            success.append(self.__directories.create(
                path=os.path.join(self.__configurations.artefacts_, pathway, code.hospital_code)
            ))

        return all(success)

    def exc(self):
        """
        The testing data has <ahead> instances.  Altogether predict <2 * ahead> points
        into the future.  The first set of ahead points are for weekly evaluations of
        a week's model; the true value of the latter set of ahead points will be known
        in future.

        :return:
        """

        # Additional delayed tasks
        decompose = dask.delayed(src.modelling.decompose.Decompose(arguments=self.__arguments).exc)
        splits = dask.delayed(src.modelling.splits.Splits(arguments=self.__arguments).exc)
        sc = dask.delayed(src.modelling.sc.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for code in self.__codes:
            """
            1. get institution data
            2. set up directories per institution
            3. decompose institution data
            4. split institution data
            5. seasonal component modelling: naive model
            """

            data: pd.DataFrame = self.__get_data(code=code)
            success: bool = self.__set_directories(code=code)
            decompositions: pd.DataFrame = decompose(data=data)
            master: mr.Master = splits(data=decompositions, code=code, success=success)
            state: bool = sc(master=master, code=code)
            computations.append(state)

        states = dask.compute(computations, scheduler='threads')
        logging.info(states)
