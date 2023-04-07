import logging


def setup_custom_logger(level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
