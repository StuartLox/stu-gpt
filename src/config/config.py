from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Contains all the mandetory configuration for
    processing the text data
    """
    path: str
    train_split: float
    block_size: int
    truncate: float


@dataclass
class OptimizerConfig:
    optimizer: str
    learning_rate: float


@dataclass
class ModelConfig:
    model: str
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


@dataclass
class TrainConfig:
    batch_size: int
    max_iters: int
    eval_steps: int
    max_epochs: int
    device: str
    grad_norm_clip: float
    save_every: int
    max_epochs: int
    snapshot_path: str
    data_loader_workers: int
    use_amp: bool


@dataclass
class Config:
    """
    Configuration object to be used in the config factory which
    brings together each configuration objects
    """
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    optimizer: OptimizerConfig
