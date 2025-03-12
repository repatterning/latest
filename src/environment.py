"""Module environment.py"""
import logging
import os

# noinspection PyUnresolvedReferences
import jax


class Environment:
    """
    <b>Notes</b><br>
    -------<br>

    Important settings include<br>
        os.environ['OMP_NUM_THREADS'] = "..."<br>
        os.environ['DP_INTRA_OP_PARALLELISM_THREADS'] = "..."<br>
        os.environ['DP_INTER_OP_PARALLELISM_THREADS'] = "..."<br>
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of model development, and supplementary, arguments.
        """

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

        self.__logger.info(arguments)

        self.__logger.info('The number of GPU devices: %s', jax.device_count(backend='gpu'))
        self.__logger.info('The number of CPU devices/cores: %s', jax.device_count(backend='cpu'))
        self.__logger.info('CPU: %s', os.cpu_count())
        self.__logger.info('Applicable Devices: %s', jax.local_device_count())

        self.__logger.info('The default device (depends on the jax.config.update setting): %s', jax.local_devices()[0])
        self.__logger.info('Active GPU: %s', str(jax.local_devices()[0]).startswith('cuda'))
