import numpy as np

import pymc
import arviz


class Conditionals:

    def __init__(self, gp: pymc.gp.Marginal, details: arviz.InferenceData, abscissae: np.ndarray):

        self.__gp = gp
        self.__details = details
        self.__abscissae = abscissae

    def __free(self, model: pymc.model.Model):

        with model:
            self.__gp.conditional('estimates', self.__abscissae, pred_noise=False)
            predictions = pymc.sample_posterior_predictive(
                self.__details, var_names=['estimates'])

    def __noisy(self, model: pymc.model.Model):

        with model:
            self.__gp.conditional('n_estimates', self.__abscissae, pred_noise=True)
            n_predictions = pymc.sample_posterior_predictive(
                self.__details, var_names=['n_estimates'])

    def exc(self, model: pymc.model.Model):
        pass
