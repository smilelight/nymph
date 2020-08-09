
import torch.nn as nn

from .base import BaseConfig, BaseModel
from .config import DEVICE, DEFAULT_CONFIG


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)


class LinearClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.class_num = args.class_num
        self.feature_dimension = args.feature_dimension
        self.window_size = args.window_size
        self.hidden_dimension = 512

        self.seq = nn.Sequential(
            nn.Linear(self.feature_dimension*self.window_size, self.hidden_dimension),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.Linear(self.hidden_dimension, self.hidden_dimension),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dimension, self.hidden_dimension),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(self.hidden_dimension, self.class_num)
        )

    def forward(self, x):
        x = x.reshape((-1, self.window_size*self.feature_dimension))
        x = self.seq(x)
        return x
