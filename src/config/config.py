import configparser
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path


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
    dropout: float

    def __post_init__(self):
        field_types = {field.name: field.type for field in fields(self)}
        for attr, value in self.__dict__.items():
            for _type in [int, bool, float, str]:
                if field_types[attr] == _type:
                    cast_value = _type(value)
                    setattr(self, attr, cast_value)


def config_from_file(section: str, into: DataConfig, file_path: str) -> Config:
    """
    Builds Config Object from Config file

    :param: section: Config section from config file
    :param: into: Config Object to build from file
    :param: file_path: File path to config.cfg file, default: config/config.cfg

    returns: Config object
    """
    config = configparser.ConfigParser()
    path = Path(file_path).resolve()
    config.read_file(open(path))
    config.sections()

    return into(**config[section])
