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

        :param code:
        :return:
        """

        return self.__data.copy().loc[self.__data['hospital_code'] == code.hospital_code, :]

    @dask.delayed
    def __set_directories(self, code: str) -> bool:
        """

        :param code:
        :return:
        """

        success = []
        for pathway in ['data', 'models']:
            success.append(
                self.__directories.create(path=os.path.join(self.__configurations.artefacts_, pathway, code)))
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
            6. trend component modelling: gaussian processes
            """

            data = self.__get_data(code=code.hospital_code)
            success = self.__set_directories(code=code.hospital_code)
            decompositions = decompose(data=data)
            master: mr.Master = splits(data=decompositions, code=code.hospital_code, success=success)
            message = sc(master=master, code=code)
            computations.append(message)

        messages = dask.compute(computations, scheduler='threads')[0]
        logging.info(messages)
