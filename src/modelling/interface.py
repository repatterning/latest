"""Module interface.py"""
import logging
import pandas as pd
import dask

import config
import src.functions.directories
import src.modelling.codes
import src.modelling.core
import src.modelling.data

class Interface:
    """
    The interface to the seasonal & trend component modelling steps.
    """

    def __init__(self, assets: pd.DataFrame, arguments: dict):
        """

        :param assets: Of assets
        :param arguments: A set of model development, and supplementary, arguments.
        """

        self.__assets = assets[:8]
        self.__arguments = arguments

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
        
        
        computations = []
        for ts_id in self.__assets['ts_id'].unique():
            
            sections = self.__get_sections(ts_id=ts_id)
            data = __get_data(sections=sections)
            computations.append(...)
            
        calculations = dask.compute(computations, scheduler='threads')[0]
        logging.info(calculations)
