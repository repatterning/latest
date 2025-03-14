"""Module main.py"""
import os
import sys

# noinspection PyUnresolvedReferences
import jax


def main():
    """

    :return:
    """

    # Setting up
    # src.preface.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

    # Data
    data = src.data.interface.Interface(s3_parameters=s3_parameters).exc()

    # Modelling
    src.modelling.interface.Interface(
      data=data, arguments=arguments).exc()

    # Transfer
    src.transfer.interface.Interface(
       connector=connector, service=service, s3_parameters=s3_parameters).exc()

    # Cache
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() - 1)

    # Classes
    import src.data.interface
    import src.functions.cache
    import src.functions.service
    import src.modelling.interface
    import src.s3.s3_parameters
    import src.s3.configurations
    import src.preface.setup
    import src.transfer.interface
    import src.preface.interface

    connector, s3_parameters, service, arguments = src.preface.interface.Interface().exc()

    # Vis-à-vis Amazon & Development: Connector, S3 Parameters, Platform Services, Configurations
    # connector = boto3.session.Session()
    # s3_parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
    # service = src.functions.service.Service(connector=connector, region_name=s3_parameters.region_name).exc()
    # arguments: dict = src.s3.configurations.Configurations(connector=connector).objects(
    #     key_name=('artefacts' + '/' + 'architecture' + '/' + 'single' + '/' + 'parts' + '/' + 'arguments.json'))

    # pytensor.config.blas__ldflags = '-llapack -lblas -lcblas'

    # jax.config.update('jax_platform_name', arguments.get('device'))
    # jax.config.update('jax_enable_x64', False if arguments.get('device') == 'gpu' else True)
    #
    # numpyro.set_platform(arguments.get('device'))
    # numpyro.set_host_device_count(
    #     jax.device_count(backend='cpu') if arguments.get('device') == 'cpu' else jax.device_count(backend='gpu'))

    # Environment Variables
    # environment.Environment(arguments=arguments)

    main()
