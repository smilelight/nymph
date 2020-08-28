import torch.nn as nn

from ..core.base import BaseConfig, BaseModel
from ..core.config import DEVICE, DEFAULT_CONFIG


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
        self.hidden_dimension = 256

        self.seq = nn.Sequential(
            nn.Linear(self.feature_dimension, self.hidden_dimension),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.Linear(self.hidden_dimension, self.hidden_dimension),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(self.hidden_dimension, self.class_num)
        ).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.seq(x)
        return x.to(DEVICE)
