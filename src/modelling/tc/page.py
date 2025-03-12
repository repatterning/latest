"""Module page.py"""
import logging
import os

import pymc


class Page:
    """
    <b>Notes</b><br>
    ------<br>

    This class saves the details of an institution's trend component model.
    """

    def __init__(self, model: pymc.model.Model, path: str):
        """

        :param model: The trend component model of an institution.
        :param path:
        """

        self.__model = model
        self.__path = path

    def __graph(self, label: str):
        """
        
        :param label: 
        :return: 
        """

        pathstr = os.path.join(self.__path, f'tcf_{label}.pdf')

        try:
            pymc.model_graph.model_to_graphviz(
                model=self.__model, figsize=(2, 2), save=pathstr, dpi=1200)
            logging.info('%s: %s', os.path.basename(self.__path), os.path.basename(pathstr))
        except IOError as err:
            raise err from err

    def __text(self, label: str):
        """
        
        :param label: 
        :return: 
        """

        pathstr = os.path.join(self.__path, f'tcf_{label}.txt')

        try:
            with open(file=pathstr, mode='w', encoding='utf-8', newline='\r\n') as disk:
                disk.write(self.__model.str_repr())
            logging.info('%s: %s', os.path.basename(self.__path), os.path.basename(pathstr))
        except IOError as err:
            raise err from err

    def exc(self, label: str):
        """

        :return:
        """

        self.__graph(label=label)
        self.__text(label=label)
