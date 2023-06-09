from typing import List
from typing import Tuple

import torch
from config.config import DataConfig
from config.config import TrainConfig
from utils.codec import Codec


class Preprocessing:
    """
    Responsible for all data preperation required before model training

    :params: data: List[data]
    """

    def __init__(self, train_conf: TrainConfig, data_conf: DataConfig):
        self.train_conf = train_conf
        self.data_conf = data_conf
        self.codec = None
        self.vocab_size = None

    def read_file(self, file_path: str) -> None:
        """
        reads file and returns raw string output

        :param: file_path - Full path where file is located
        :returns Full text of raw file
        """
        with open(file_path) as f:
            self.full_text = f.read()

    def encode_text(self):
        self.codec = Codec(self.full_text)
        self.vocab_size = self.codec.vocab_size
        self.data = torch.tensor(self.codec.encode(self.full_text), dtype=torch.long)

    def train_val_split(self) -> None:
        """
        Split up the training data into train and validation data

        data: List[int]: List of text encoded integers
        """
        n = int(len(self.data) * self.data_conf.train_split)
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split: str) -> Tuple[List[int], List[int]]:
        """
        Generate a small batch of training data of inputs x and targets y

        :param: split: dictionary of training or test data
        """
        if split == 'train':
            data = self.train_data

        if split == 'val':
            data = self.val_data

        batch_size = self.train_conf.batch_size
        block_size = self.data_conf.block_size

        start = len(data) - block_size
        ix = torch.randint(start, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[(i + 1):(i + block_size + 1)] for i in ix])
        return x, y
