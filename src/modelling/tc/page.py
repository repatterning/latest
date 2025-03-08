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
        pass