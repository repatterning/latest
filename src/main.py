"""Module main.py"""
import os
import sys


def main():
    """

    :return:
    """

    # Data
    src.data.interface.Interface(service=service, s3_parameters=s3_parameters, arguments=arguments).exc()

    # Modelling
    # src.modelling.interface.Interface(
    #   data=data, arguments=arguments).exc()

    # Transfer
    # src.transfer.interface.Interface(
    #    connector=connector, service=service, s3_parameters=s3_parameters).exc()

    # Cache
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Classes
    import src.data.interface
    import src.functions.cache
    import src.modelling.interface
    import src.transfer.interface
    import src.preface.interface

    connector, s3_parameters, service, arguments = src.preface.interface.Interface().exc()

    main()
