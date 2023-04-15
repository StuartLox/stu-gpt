import os
from typing import Any
from typing import Tuple

import hydra
import torch
import torch.nn as nn
from config.config import Config
from config.config import DataConfig
from config.config import ModelConfig
from config.config import OptimizerConfig
from config.config import TrainConfig
from exceptions import ModelNotFoundError
from exceptions import OptimizerNotFoundError
from impl.preprocessing import Preprocessing
from model.bigram import BigramLangaugeModel
from model.gpt import GPTLanguageModel
from omegaconf import DictConfig
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group
from torch.utils.data import random_split
from trainer import Trainer
from utils.charset import CharDataset
from utils.custom_logger import setup_custom_logger


torch.manual_seed(1337)
module = __name__
logger = setup_custom_logger(module)


def ddp_setup(train_conf: TrainConfig):
    logger.info(f"GPU CUDA Enabled - {torch.cuda.is_available()}")
    if train_conf.device == "gpu":
        backend = 'nccl'
    else:
        backend = 'gloo'

    init_process_group(backend=backend, world_size=1, rank=0)


def config_factory(cfg: DictConfig) -> Config:
    """
    Factory function to turn Dict into Config Object. Where the
    configurations are composed of Specific Config types (i.e data, train, model, and optimizer)

    :param cfg: Hydra configuration object
    :returns Config: Complete Configuration Object
    """
    data = DataConfig(**cfg['data'])
    optimizer = OptimizerConfig(**cfg['optimizer'])
    model = ModelConfig(**cfg['model'])
    train = TrainConfig(**cfg['train'])

    return Config(data=data, optimizer=optimizer, model=model, train=train)


def preprocessing_factory(config: Config) -> Preprocessing:
    """
    Factory method responsible for performing all preprocessing
    logic before model training

    :returns: preprocessing: contains batches, training, and test data
    """
    preprocessing = Preprocessing(data_conf=config.data, train_conf=config.train)
    preprocessing.read_file(config.data.path)
    preprocessing.encode_text()
    preprocessing.train_val_split()
    return preprocessing


def model_factory(preprocessing: Preprocessing, config: Config):
    """
    Responsible for building the model object

    :param model_name: model used for learning
    :returns model: Contructed model to be used for training
    :raises ModelNotFoundError: Exception where selected model is not available
    """
    model_map = {
        'bigram': BigramLangaugeModel,
        'gpt': GPTLanguageModel,
    }
    if model := model_map.get(config.model.model):
        logger.info(f"Model {model.__name__} selected")
        return model(preprocessing.vocab_size, config)

    logger.error(f"Model {config.model} not found")
    raise ModelNotFoundError


def optimizer_factory(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """
    Factory function to construct the pytorch optimizer for the model

    :param model: Model object passed into the optimizer
    :param config: Configuration object containing the hyperparams
    :returns: pytorch Optimizer object used for optimization
    :raises OptimizerNotFoundError:
    """
    otimizer_map = {
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'ada_delta': torch.optim.Adadelta,
    }
    if optimizer := otimizer_map.get(config.optimizer):
        logger.info(f"Selected Optimizer {config.optimizer} with learning rate: {config.learning_rate}")
        return optimizer(model.parameters(), lr=config.learning_rate)

    logger.error(f"Optimizer {config.optimizer} not found from Config")
    raise OptimizerNotFoundError


def get_train_objs(data_cfg: DataConfig) -> Tuple[Any]:
    """
    Gets the required Training Objects for model training

    :param:
    """
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    return train_set, test_set


def get_inference(preprocessing: Preprocessing, model: nn.Module, tokens: int):
    context = torch.zeros((1, 1), dtype=torch.long)
    print(preprocessing.codec.decode(model.generate(context, max_new_tokens=tokens)[0].tolist()))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    config = config_factory(cfg)
    ddp_setup(config.train)

    preprocessing = preprocessing_factory(config=config)

    model = model_factory(
        preprocessing=preprocessing,
        config=config,
    )
    train_data, test_data = get_train_objs(data_cfg=config.data)
    optimizer = optimizer_factory(model=model, config=config.optimizer)

    trainer = Trainer(
        config=config.train,
        local_rank=int(os.environ['LOCAL_RANK']),
        global_rank=int(os.environ['RANK']),
        model=model,
        optimizer=optimizer,
        train_dataset=train_data,
        test_dataset=test_data,
    )
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main()
