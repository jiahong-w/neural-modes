import time
import logging

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, desc=None, verbose=False):
        self.desc = desc
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, value, traceback):
        self.end = time.time()
        duration_ms = (self.end - self.start) * 1000.0
        if self.verbose:
            logger.debug(f'Timing {self.desc}: {duration_ms:.4f} ms')