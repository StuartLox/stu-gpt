import torch
import torch.nn as nn

from impl.preprocessing import Preprocessing
from config.config import DataConfig, config_from_file
from utils.custom_logger import setup_custom_logger
from model.bigram import BigramLangaugeModel
from model.gpt import GPTLanguageModel
from src.exceptions import (
    ModelNotFoundError,
    OptimizerNotFoundError,
)


torch.manual_seed(1337)
logger = setup_custom_logger()


def data_config_factory(file_path: str = 'config/config.cfg') -> DataConfig:
    """
    Factory method which builds data config object
    :returns: pre-built factory
    """
    return config_from_file(
        section='data',
        into=DataConfig,
        file_path=file_path
    )


def preprocessing_factory(config: DataConfig) -> Preprocessing:
    """
    Factory method responsible for performing all preprocessing
    logic before model training

    :returns: preprocessing: contains batches, training, and test data
    """
    preprocessing = Preprocessing(config=config)
    preprocessing.read_file(config.training_file)
    preprocessing.encode_text()
    preprocessing.train_val_split()
    return preprocessing


def model_factory(preprocessing: Preprocessing, config: DataConfig):
    """
    Responsible for building the model object from the availble deep learning models

    :param model_name: model used for learning
    :returns model: Contructed model to be used for training
    :raises ModelNotFoundError: Exception where selected model is not available
    """
    model_map = {
        'bigram': BigramLangaugeModel,
        'gpt': GPTLanguageModel,
    }
    if model := model_map.get(config.model):
        logger.info(f"Model {model.__name__} selected")
        return model(preprocessing.vocab_size, config)

    logger.error(f"Model {config.model} not found")
    raise ModelNotFoundError


def optimizer_factory(model: nn.Module, config: DataConfig) -> torch.optim.Optimizer:
    """
    Factory function to construct the pytorch optimizer for the model. Takes in a Config Object
    to set hyper params.

    :param model: Model object passed into the optimizer
    :param config: Configuration object containing the hyperparams (i.e learning rate)
    :returns: pytorch Optimizer object used for optimization
    :raises OptimizerNotFoundError: Exception when none of the accepted optimizers are passed into the config
    """
    otimizer_map = {
        'adamw':  torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'ada_delta': torch.optim.Adadelta,
    }
    if optimizer := otimizer_map.get(config.optimizer):
        logger.info(f"Selected Optimizer {config.optimizer} with learning rate: {config.learning_rate}")
        return optimizer(model.parameters(), lr=float(config.learning_rate))
    
    logger.error(f"Optimizer {config.optimizer} not found from Config")
    raise OptimizerNotFoundError


@torch.no_grad()
def estimate_loss(preprocessing: Preprocessing, model: nn.Module, eval_steps: int, curr_iter: int):
    """
    Averages out the estimiation of the training and validation set to get an approximiate
    estimation of the training data

    :param preprocessing: Object containing the data required for model training
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


def train(
        preprocessing: Preprocessing, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        config: DataConfig
    ):
    """
    Generic execution of model training and validation of a given model

    :param preprocessing: Object containing the data required for model training
    :param model: The Pytorch model used for traning
    :param optimizer: Optimizer to be used with the model
    :config DataConfig: Model and data configuration used for traning  
    """
    for iter in range(int(config.steps)):
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
            curr_iter=iter
        )
    return model


def get_inference(preprocessing: Preprocessing, model: nn.Module, tokens: int):
    context = torch.zeros((1, 1), dtype=torch.long)
    print(preprocessing.codec.decode(model.generate(context, max_new_tokens=500)[0].tolist()))




def main():
    data_config = data_config_factory()
    preprocessing = preprocessing_factory(config=data_config)
    model_obj = model_factory(
        preprocessing=preprocessing,
        config=data_config,
    )
    optimizer = optimizer_factory(model=model_obj, config=data_config)
    
    model = train(
        preprocessing=preprocessing, 
        model=model_obj,
        optimizer=optimizer, 
        config=data_config
    )

    # Get inference from model
    get_inference(
        preprocessing=preprocessing, 
        model=model,
        tokens=500
    )




if __name__ == "__main__":
    main()
