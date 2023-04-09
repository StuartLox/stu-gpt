import logging


def setup_custom_logger(module: str, level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(module)
