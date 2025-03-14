"""Module initial.py"""
import dask
import pandas as pd
import numpy as np

import src.elements.codes as ce
import src.elements.master as mr
import src.functions.directories
import src.modelling.decompose
import src.modelling.sc.interface
import src.modelling.split


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
        frame['ln'] = np.log(frame['n_attendances'])

        return frame

    def exc(self) -> list[mr.Master]:
        """

        :return:
        """

        # Additional delayed tasks
        decompose = dask.delayed(src.modelling.decompose.Decompose(arguments=self.__arguments).exc)
        split = dask.delayed(src.modelling.split.Split(arguments=self.__arguments).exc)
        sc = dask.delayed(src.modelling.sc.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for code in self.__codes:

            data: pd.DataFrame = self.__get_data(code=code)
            master: mr.Master = split(data=data, code=code)
            _master: mr.Master = decompose(master=master, code=code)
            master_: mr.Master | None = sc(master=_master, code=code)
            computations.append(master_)

        masters = dask.compute(computations, scheduler='threads')[0]
        masters = [master for master in masters if master is not None]

        return masters
