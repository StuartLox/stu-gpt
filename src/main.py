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
from utils.custom_logger import setup_custom_logger


torch.manual_seed(1337)
module = __name__
logger = setup_custom_logger(module)


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
    preprocessing.read_file(config.data.training_file)
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


@torch.no_grad()
def estimate_loss(preprocessing: Preprocessing, model: nn.Module, eval_steps: int, curr_iter: int):
    """
    Averages out the estimiation of the training and validation set to get an
    approximiate estimation of the training data

    :param preprocessing: Object containing the data for model training
    :param model: The Pytorch model used for traning
    :config DataConfig: Contains evalution
    """
    if curr_iter % eval_steps != 0:
        return

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = preprocessing.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    logger.info(f"step {curr_iter}: train loss {out['train']:.4f}, val loss {out['val']}")


def train(preprocessing: Preprocessing, model: nn.Module, optimizer: torch.optim.Optimizer, config: DataConfig):
    """
    Generic execution of model training and validation of a given model

    :param preprocessing: Object containing the data required for model training
    :param model: The Pytorch model used for traning
    :param optimizer: Optimizer to be used with the model
    :param config: Model and data configuration used for traning
    """
    for iter in range(int(config.max_iters)):
        # sample a batch of data
        xb, yb = preprocessing.get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        estimate_loss(
            preprocessing=preprocessing,
            model=model,
            eval_steps=int(config.eval_steps),
            curr_iter=iter,
        )
    return model


def get_inference(preprocessing: Preprocessing, model: nn.Module, tokens: int):
    context = torch.zeros((1, 1), dtype=torch.long)
    print(preprocessing.codec.decode(model.generate(context, max_new_tokens=tokens)[0].tolist()))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = config_factory(cfg)
    preprocessing = preprocessing_factory(config=config)

    model_obj = model_factory(
        preprocessing=preprocessing,
        config=config,
    )

    optimizer = optimizer_factory(model=model_obj, config=config.optimizer)

    model = train(
        preprocessing=preprocessing,
        model=model_obj,
        optimizer=optimizer,
        config=config.train,
    )

    print(model)

    # # Get inference from model
    # get_inference(
    #     preprocessing=preprocessing,
    #     model=model,
    #     tokens=500,
    # )


if __name__ == "__main__":
    main()
