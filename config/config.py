import configparser
from dataclasses import dataclass


class Config:
    pass


@dataclass
class DataConfig(Config):
    """
    Contains all the mandetory configuration for
    processing the text data
    """
    training_file: str
    batch_size: int
    block_size: int
    train_val_split: float
    optimizer: str
    learning_rate: float
    model: str
    max_iters: int
    eval_steps: int
    device: str
    n_embd: int
    n_head: int
    n_layer: int
    dropout: int


def config_from_file(section: str, into: Config, file_path: str) -> Config:
    """
    Builds Config Object from Config file

    :param: section: Config section from config file
    :param: into: Config Object to build from file
    :param: file_path: File path to config.cfg file, default: config/config.cfg

    returns: Config object
    """
    config = configparser.ConfigParser()
    config.read_file(open(file_path))
    config.sections()

    return into(**config[section])


if __name__ == "__main__":
    data_conf = config_from_file(
        section='data',
        into=DataConfig,
        file_path='./config/config.cfg'
    )
    print(data_conf)
