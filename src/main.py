"""Module main.py"""
import logging
import os
import sys


def main():
    """

    :return:
    """

    # Logging
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info(__name__)

    # Data
    assets = src.assets.interface.Interface(service=service, s3_parameters=s3_parameters, arguments=arguments).exc()
    logger.info(assets)

    # Modelling
    src.modelling.interface.Interface(assets=assets, arguments=arguments).exc()

    # Transfer
    src.transfer.interface.Interface(
       connector=connector, service=service, s3_parameters=s3_parameters).exc()

    # Cache
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Classes
    import src.assets.interface
    import src.functions.cache
    import src.modelling.interface
    import src.transfer.interface
    import src.preface.interface

    connector, s3_parameters, service, arguments = src.preface.interface.Interface().exc()

    main()
