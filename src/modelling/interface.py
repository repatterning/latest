"""Module interface.py"""
import logging

import dask
import pandas as pd

import src.elements.partitions as pr
import src.elements.master as mr
import src.modelling.data
import src.modelling.split
import src.modelling.architecture.interface


class Interface:
    """
    The interface to the seasonal & trend component modelling steps.
    """

    def __init__(self, listings: pd.DataFrame, arguments: dict):
        """

        :param listings: List of files
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__listings = listings
        self.__arguments = arguments

    @dask.delayed
    def __get_listing(self, ts_id: int) -> list[str]:
        """

        :param ts_id:
        :return:
        """

        return self.__listings.loc[
            self.__listings['ts_id'] == ts_id, 'uri'].to_list()

    def exc(self, partitions: list[pr.Partitions]):
        """
        Via dask dataframe, read-in a machine's set of measures files.  Subsequently, split, then add the
        relevant features to the training data split.

        :return:
        """

        __get_data = dask.delayed(src.modelling.data.Data(arguments=self.__arguments).exc)
        __get_splits = dask.delayed(src.modelling.split.Split(arguments=self.__arguments).exc)
        __architecture = dask.delayed(src.modelling.architecture.interface.Interface(arguments=self.__arguments).exc)

        computations = []
        for partition in partitions:

            listing = self.__get_listing(ts_id=partition.ts_id)
            data = __get_data(listing=listing)
            master: mr.Master = __get_splits(data=data, partition=partition)
            message = __architecture(master=master, partition=partition)
            computations.append(message)

        messages = dask.compute(computations, scheduler='threads')[0]
        logging.info(messages)
