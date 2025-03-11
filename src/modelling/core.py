import logging
import os
import glob
import dask
import pandas as pd

import config
import src.elements.codes as ce
import src.functions.streams
import src.elements.text_attributes as txa
import src.modelling.tc.interface


class Core:

    def __init__(self, codes: list[ce.Codes], arguments: dict):
        """
        
        :param codes: 
        :param arguments: 
        """

        self.__codes = codes
        self.__arguments = arguments

        self.__configurations = config.Config()
        self.__streams = src.functions.streams.Streams()
        
    def __get_codes(self) -> list[ce.Codes]:
        """

        :return:
        """

        strings = glob.glob(
            pathname=os.path.join(self.__configurations.artefacts_, 'data', '**', '*training.csv'))
        values = [os.path.basename(os.path.dirname(string)) for string in strings]
        
        return [code for code in self.__codes if code.hospital_code in values]

    @dask.delayed
    def __get_training_data(self, code: ce.Codes) -> pd.DataFrame:
        """

        :param code:
        :return:
        """

        uri = os.path.join(self.__configurations.artefacts_, 'data', code.hospital_code, 'training.csv')
        text = txa.TextAttributes(uri=uri, header=0)
        
        return self.__streams.read(text=text)

    def exc(self) -> list[str]:
        """

        :return:
        """

        tc = dask.delayed(src.modelling.tc.interface.Interface(arguments=self.__arguments).exc)

        codes = self.__get_codes()

        computations = []
        for code in codes:            
            training = self.__get_training_data(code=code)
            message = tc(training=training, code=code)
            computations.append(message)

        messages = dask.compute(computations, num_workers=4)[0]

        return messages
