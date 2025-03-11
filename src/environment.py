import logging
import os

import jax
import numpyro


class Environment:

      def __init__(self):
          """
          Later, set by equations.  If CPU
            intra == # of MCMC chains
            omp == intra
            inter = floor(os.cpu_count() / intra)
          subject to counts constraints.
          """

          os.environ['XLA_FLAGS'] = (
              '--xla_disable_hlo_passes=constant_folding '
              f'--xla_force_host_platform_device_count={os.cpu_count()} ')
          os.environ['OMP_NUM_THREADS'] = "4"
          os.environ['DP_INTRA_OP_PARALLELISM_THREADS'] = "4"
          os.environ['DP_INTER_OP_PARALLELISM_THREADS'] = "4"

          # Logging
          logging.basicConfig(level=logging.INFO,
                              format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                              datefmt='%Y-%m-%d %H:%M:%S')
          self.__logger = logging.getLogger(__name__)

      def __call__(self, arguments: dict):
          """

          :param arguments:
          :return:
          """

          jax.config.update('jax_platform_name', arguments.get('device'))
          jax.config.update('jax_enable_x64', False if arguments.get('device')  == 'gpu' else True)

          numpyro.set_platform(arguments.get('device'))
          numpyro.set_host_device_count(os.cpu_count() if arguments.get('device') == 'cpu' else jax.device_count(backend='gpu'))

          self.__logger.info('The number of GPU devices: %s', jax.device_count(backend='gpu'))
          self.__logger.info('The number of CPU devices/cores: %s', jax.device_count(backend='cpu'))

          self.__logger.info('The default device (depends on the jax.config.update setting): %s', jax.local_devices()[0])
          self.__logger.info('Active GPU: %s', str(jax.local_devices()[0]).startswith('cuda'))
