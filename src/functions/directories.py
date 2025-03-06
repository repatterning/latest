"""
Module: directories
"""
import os
import logging


class Directories:
    """
    For clearing & creating directories
    """

    def __init__(self):
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

        self.__logger: logging.Logger = logging.getLogger(__name__)


    def cleanup(self, path: str) -> bool:
        """
        Clears a directory
        :param path:
        :return:
        """

        # Does the directory exist?
        if not os.path.exists(path=path):
            return True

        # If the directory exists, delete the files
        __files = [os.remove(os.path.join(base, file))
                   for base, _, files in os.walk(path) for file in files]
        self.__logger.info(__files)

        # ... inspect
        elements = [file for _, _, files in os.walk(path) for file in files]
        assert len(elements) == 0, f'Unable to delete all files within path {path}'

        # ... then the directories
        __directories = [os.removedirs(os.path.join(base, directory))
                         for base, directories, _ in os.walk(path, topdown=False)
                         for directory in directories
                         if os.path.exists(os.path.join(base, directory))]
        self.__logger.info(__directories)

        # ... and inspect
        elements = [directory for _, directories, _ in os.walk(path) for directory in directories]
        assert len(elements) == 0, f'Unable to delete all directories within path {path}'

        # Hence
        return len(elements) == 0

    @staticmethod
    def create(path: str) -> bool:
        """
        :param path:
        :return:
        """

        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return True
        except OSError as err:
            raise err
