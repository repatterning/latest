"""Module interface.py"""
import logging
import os
import sys

import pandas as pd

import config
import src.elements.codes as ce
import src.functions.directories
import src.modelling.codes
import src.modelling.core
import src.modelling.decompose
import src.modelling.sc.interface
import src.modelling.splits
import src.modelling.initial


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

        # Instances
        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()

    def __set_directories(self, codes: list[ce.Codes]):
        """

        :param codes:
        :return:
        """

        directories = [self.__directories.create(os.path.join(self.__configurations.artefacts_, section, c.hospital_code))
                       for section in ['data', 'models'] for c in codes]

        if not all(directories):
            sys.exit('Missing Directories')

    def exc(self):
        """
        Each instance of codes consists of the health board & institution/hospital codes of an institution/hospital.
        
        :return: 
        """

        # Codes
        codes: list[ce.Codes] = src.modelling.codes.Codes().exc(data=self.__data)
        codes = codes[:4]

        # Directories
        self.__set_directories(codes=codes)

        # Modelling
        states = src.modelling.initial.Initial(
            data=self.__data, codes=codes, arguments=self.__arguments).exc()

        if all(states):
            message = src.modelling.core.Core(codes=codes, arguments=self.__arguments).exc()
            logging.info(message)
