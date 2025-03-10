import logging

import dask
import pandas as pd

import src.elements.codes as ce
import src.elements.master as mr
import src.functions.directories
import src.modelling.decompose
import src.modelling.sc.interface
import src.modelling.splits


class Initial:

    def __init__(self, data: pd.DataFrame, codes: list[ce.Codes], arguments: dict):
        """

        :param data:
        :param codes:
        :param arguments:
        """

        self.__data = data
        self.__codes = codes
        self.__arguments = arguments

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

    def exc(self) -> list[bool]:
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
            2. decompose institution data
            3. split institution data
            4. seasonal component modelling: naive model
            """

            data: pd.DataFrame = self.__get_data(code=code)
            decompositions: pd.DataFrame = decompose(data=data)
            master: mr.Master = splits(data=decompositions, code=code)
            state: bool = sc(master=master, code=code)
            computations.append(state)

        states = dask.compute(computations, scheduler='threads')[0]

        return states
