import torch
import torch.nn as nn
from torchcrf import CRF

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
        ).to(DEVICE)

    def forward(self, x):
        x = x.reshape((-1, self.window_size*self.feature_dimension))
        x = self.seq(x)
        return x


class BiLstmCrfClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.class_num = args.class_num
        self.feature_dimension = args.feature_dimension
        self.window_size = args.window_size
        self.hidden_dimension = 256
        self.batch_size = args.batch_size

        self.num_layers = 2

        self.lstm = nn.LSTM(self.feature_dimension, self.hidden_dimension // 2, num_layers=self.num_layers, dropout=0.5,
                            bidirectional=True).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dimension, self.class_num).to(DEVICE)

        self.crf_layer = CRF(self.class_num).to(DEVICE)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dimension // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dimension // 2).to(DEVICE)

        return h0, c0

    def forward(self, x):
        emissions = self.lstm_forward(x)
        return self.crf_layer.decode(emissions)

    def loss(self, x, y):
        emissions = self.lstm_forward(x)
        return self.crf_layer(emissions, y)

    def lstm_forward(self, x):
        hidden = self.init_hidden(batch_size=x.shape[1])
        lstm_out, _ = self.lstm(x, hidden)
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)
