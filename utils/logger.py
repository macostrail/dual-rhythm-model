import logging


def setup(LOG_LEVEL):
    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                        format=formatter)

    logger = logging.getLogger(__name__)
    return logger
