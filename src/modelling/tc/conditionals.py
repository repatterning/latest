
import pymc
import arviz


class Conditionals:

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData):

        self.__gp = gp
        self.__details = details

    def exc(self):
        pass
