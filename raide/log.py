from loguru import logger
import sys
def filter_log_record(record):
    blacklist = ["fish_speech", "RealtimeSTT"]
    return not any(black in record["name"] for black in blacklist)

def init_logger(suppress_other_log: bool = True):
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "- <level>{message}</level>"
    )

    logger.remove()  # Remove default logger
    logger.add(sys.stdout, format=log_format, filter=filter_log_record, level="INFO")

