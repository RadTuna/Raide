from contextlib import contextmanager
import time
from loguru import logger

@contextmanager
def profile(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"Profile '{name}': {end - start:.4f} seconds")
