"""Module interface.py"""
import logging
import pandas as pd
import dask

import config
import src.elements.master as mr
import src.functions.directories
import src.modelling.core
import src.modelling.data
import src.modelling.gauges
import src.modelling.split


class Interface:
    """
    The interface to the seasonal & trend component modelling steps.
    """

    def __init__(self, assets: pd.DataFrame, arguments: dict):
        """

        :param assets: Of assets
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__assets = assets
        self.__arguments = arguments

        # The gauges
        self.__gauges = src.modelling.gauges.Gauges().exc(assets=assets)

        # Instances
        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()

    @dask.delayed
    def __get_sections(self, ts_id: int) -> list:

        return self.__assets.loc[
            self.__assets['ts_id'] == ts_id, 'uri'].to_list()

    def exc(self):
        """
        Via dask dataframe, read-in a machine's set of measures files.  Subsequently, split, then add the
        relevant features to the training data split.

        :return:
        """
        
        __get_data = dask.delayed(src.modelling.data.Data().exc)
        __get_splits = dask.delayed(src.modelling.split.Split(arguments=self.__arguments).exc)
        
        
        computations = []
        for gauge in self.__gauges:
            
            sections = self.__get_sections(ts_id=gauge.ts_id)
            data = __get_data(sections=sections)
            master: mr.Master = __get_splits(data=data, gauge=gauge)
            computations.append(...)
            
        calculations = dask.compute(computations, scheduler='threads')[0]
        logging.info(calculations)
