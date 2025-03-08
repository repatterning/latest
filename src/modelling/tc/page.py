
import pymc
import src.elements.codes as ce

class Page:

    def __init__(self, model: pymc.model.Model, code: ce.Codes):

        self.__model = model
        self.__code = code

    def __graph(self):
        pass

    def __text(self):
        pass