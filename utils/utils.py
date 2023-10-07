
from time import time
import logging
import os
logger = logging.getLogger('utils')

logger.setLevel(logging.DEBUG)

if not os.path.exists('./logs'):
    os.makedirs('./logs')

handler = logging.FileHandler('./logs/times.log')

handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def compute_size(f):
    """computes the time required for a function to run

    Args:
        f (function): function to be computed
    """
    def wrapper(*args, **kwargs):
        start = time()
        var = f(*args, **kwargs)
        logger.debug(
            f"Time required for function {f.__name__}: {time() - start} seconds")
        return var
    return wrapper
