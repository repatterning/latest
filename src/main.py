"""Module main.py"""
import logging
import os
import sys

import boto3
import jax
import numpyro
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

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Classes
    import src.functions.service
    import src.s3.s3_parameters
    import src.s3.configurations

    # Vis-à-vis Amazon & Development: Connector, S3 Parameters, Platform Services, Configurations
    connector = boto3.session.Session()
    s3_parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
    service = src.functions.service.Service(connector=connector, region_name=s3_parameters.region_name).exc()
    arguments: dict = src.s3.configurations.Configurations(connector=connector).objects(
        key_name=('artefacts' + '/' + 'architecture' + '/' + 'single' + '/' + 'arguments.json'))

    # Environment Variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_FLAGS'] = (
        '--xla_disable_hlo_passes=constant_folding '
    )
    os.environ['OMP_NUM_THREADS'] = "1"
    pytensor.config.blas__ldflags = '-llapack -lblas -lcblas'
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_enable_x64', False)
    # numpyro.set_host_device_count(12 if arguments.get('device') == 'cpu' else jax.device_count(backend='gpu'))
    numpyro.set_platform('gpu')

    # Classes
    import src.data.interface
    import src.functions.cache
    import src.modelling.core
    import src.modelling.interface
    import src.setup
    import src.transfer.interface

    main()
