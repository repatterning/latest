import numpy as np

import pymc
import arviz


class Conditionals:

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray):

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae

    def __execute(self, name: str, model_: pymc.model.Model, pred_noise: bool):

        with model_:
            self.__gp.conditional(name=name, Xnew=self.__abscissae, pred_noise=pred_noise)
            objects = pymc.sample_posterior_predictive(self.__details, var_names=[name])

        return model_, objects

    def exc(self, model: pymc.model.Model):
        pass



