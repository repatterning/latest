import os
import pymc

import config
import src.elements.codes as ce

class Page:

    def __init__(self, model: pymc.model.Model, code: ce.Codes):
        """

        :param model:
        :param code:
        """

        self.__model = model
        self.__code = code

        configurations = config.Config()
        self.__root = os.path.join(configurations.artefacts_, 'models', code.hospital_code)

    def __graph(self):
        pass

    def __text(self):

        pathstr = os.path.join(self.__root, 'tcf_algorithm.txt')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__model.str_repr())
        except IOError as err:
            raise err from err
