# -*- coding: utf-8 -*-
import math

from torch.utils.data import Dataset, random_split


def get_split_size(total_len: int, train_ratio: float = 0.7, dev_ratio: float = 0.2, test_ratio: float = 0.1):
    assert math.isclose(sum([train_ratio, dev_ratio, test_ratio]), 1.0, rel_tol=1e-5)
    train_size = round(total_len * train_ratio)
    dev_size = round(total_len * dev_ratio)
    test_size = total_len - train_size - dev_size
    return train_size, dev_size, test_size


def split_dataset(dataset: Dataset, sizes: tuple = (0.7, 0.2, 0.1)):
    assert len(dataset) > 0, "数据集长度必须要大于0"
    assert math.isclose(sum(sizes), 1.0, rel_tol=1e-5), "数据集切分比例之和应该为1"
    if len(sizes) == 3:
        train_size = round(len(dataset) * sizes[0])
        dev_size = round(len(dataset) * sizes[1])
        test_size = len(dataset) - train_size - dev_size
        return random_split(dataset, lengths=(train_size, dev_size, test_size))
    elif len(sizes) == 2:
        train_size = round(len(dataset) * sizes[0])
        test_size = len(dataset) - train_size
        return random_split(dataset, lengths=(train_size, test_size))
