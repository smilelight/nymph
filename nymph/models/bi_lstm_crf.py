import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..core.base import BaseConfig, BaseModel
from ..core.config import DEVICE, DEFAULT_CONFIG
from .tool import get_mask


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)


class BiLstmCrfClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.class_num = args.class_num
        self.feature_dimension = args.feature_dimension
        self.hidden_dimension = 256
        self.batch_size = args.batch_size

        self.num_layers = 2

        self.lstm = nn.LSTM(self.feature_dimension, self.hidden_dimension // 2, num_layers=self.num_layers, dropout=0.5,
                            bidirectional=True).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dimension, self.class_num).to(DEVICE)

        self.crf_layer = CRF(self.class_num).to(DEVICE)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dimension // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dimension // 2).to(DEVICE)

        return h0, c0

    def forward(self, x, seq_lens):
        mask = get_mask(seq_lens)
        emissions = self.lstm_forward(x, seq_lens)
        return self.crf_layer.decode(emissions, mask=mask)

    def loss(self, x, seq_lens, y):
        mask = get_mask(seq_lens)
        emissions = self.lstm_forward(x, seq_lens)
        return self.crf_layer(emissions, torch.LongTensor(y), mask=mask)

    def lstm_forward(self, x, seq_lens):
        x = pack_padded_sequence(x, seq_lens)
        hidden = self.init_hidden(batch_size=len(seq_lens))
        lstm_out, _ = self.lstm(x, hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)
