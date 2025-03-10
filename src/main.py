"""Module main.py"""
import logging
import os
import sys

import boto3
import jax
import pytensor


def main():
    """

    :return:
    """

    logger: logging.Logger = logging.getLogger(__name__)
    logger.info('The number of GPU devices: %s', jax.device_count(backend='gpu'))
    logger.info('The number of CPU devices: %s', jax.device_count(backend='cpu'))
    logger.info('The default device (depends on the jax.config.update setting): %s', jax.local_devices()[0])
    logger.info('GPU: %s', str(jax.local_devices()[0]).startswith('cuda'))
    logging.info('BLAS: %s', pytensor.config.blas__ldflags)

    # Setting up
    src.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

    # Data
    data = src.data.interface.Interface(s3_parameters=s3_parameters).exc()

    # Modelling
    src.modelling.interface.Interface(
        data=data, arguments=arguments).exc()

    # Transfer
    src.transfer.interface.Interface(connector=connector, service=service, s3_parameters=s3_parameters).exc()

    # Cache
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Environment Variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_FLAGS'] = (
        '--xla_disable_hlo_passes=constant_folding '
    )
    pytensor.config.blas__ldflags = '-llapack -lblas -lcblas'
    jax.config.update('jax_platform_name', 'gpu')

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Classes
    import src.data.interface
    import src.functions.cache
    import src.functions.service
    import src.modelling.interface
    import src.s3.configurations
    import src.s3.s3_parameters
    import src.setup
    import src.transfer.interface

    # Amazon: Connector, S3 Parameters, Service
    connector = boto3.session.Session()
    s3_parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
    service = src.functions.service.Service(connector=connector, region_name=s3_parameters.region_name).exc()

    # Modelling arguments
    arguments = src.s3.configurations.Configurations(connector=connector).objects(
        key_name=('artefacts' + '/' + 'architecture' + '/' + 'single' + '/' + 'arguments.json'))

    main()
