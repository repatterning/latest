"""Module initial.py"""
import dask
import pandas as pd

import src.elements.codes as ce
import src.elements.master as mr
import src.functions.directories
import src.modelling.decompose
import src.modelling.sc.interface
import src.modelling.splits


class Initial:
    """
    Seasonal component modelling.
    """

    def __init__(self, data: pd.DataFrame, codes: list[ce.Codes], arguments: dict):
        """

        :param data: The weekly accidents & emergency data of institutions/hospitals
        :param codes: The unique set of health board & institution pairings.
        :param arguments: A set of model development, and supplementary, arguments.
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

    def exc(self) -> list[mr.Master]:
        """
        1. get institution data
        2. decompose institution data
        3. split institution data
        4. seasonal component modelling: naive model

        :return:
        """

        # Additional delayed tasks
        decompose = dask.delayed(src.modelling.decompose.Decompose(arguments=self.__arguments).exc)
        splits = dask.delayed(src.modelling.splits.Splits(arguments=self.__arguments).exc)
        sc = dask.delayed(src.modelling.sc.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for code in self.__codes:

            data: pd.DataFrame = self.__get_data(code=code)
            decompositions: pd.DataFrame = decompose(data=data)
            master: mr.Master = splits(data=decompositions, code=code)
            master_: mr.Master | None = sc(master=master, code=code)
            computations.append(master_)

        masters = dask.compute(computations, scheduler='threads')[0]
        masters = [master for master in masters if master is not None]

        return masters
