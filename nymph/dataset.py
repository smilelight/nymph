
import torch
from torch.utils.data import Dataset

from .data_preprocessor import DataPreprocessor


class ClsDataset(Dataset):
    def __init__(self, data_preprocessor: DataPreprocessor, x_raw, y_raw, seq=False):
        self.data_preprocessor = data_preprocessor
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.x = self.data_preprocessor.transform_feature(x_raw)
        if seq:
            self.y = self.data_preprocessor.transform_targets(y_raw)
        else:
            self.y = self.data_preprocessor.transform_target(y_raw)

    def __getitem__(self, item):
        return torch.as_tensor(self.x[item]), torch.as_tensor(self.y[item], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def extend(self, other):
        assert isinstance(other, ClsDataset)
        self.x = torch.cat((self.x, other.x))
        self.y = torch.cat((self.y, other.y))
        return self




