# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Callable
from types import FunctionType

import pandas as pd
from torch.utils.data import Dataset


class NormDataset(Dataset):
    def __init__(self, dataset: Union[List[Dict], pd.DataFrame]):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict('records')
        self._dataset = dataset

    def __getitem__(self, item):
        return self._dataset[item]

    def __len__(self):
        return len(self._dataset)

    @property
    def raw_dataset(self):
        return self._dataset


def _get_span_list(idx_list: List[int], max_len: int = 30, min_len: int = 10, overlap: bool = False):
    assert len(idx_list) > 0
    res_list = []
    if overlap:
        for i in range(len(idx_list) - 1):
            flag = False
            for j in range(i + 1, len(idx_list)):
                if min_len <= idx_list[j] - idx_list[i] <= max_len:
                    res_list.append((idx_list[i], idx_list[j]))
                    flag = True
            if not flag:
                res_list.append((idx_list[i], idx_list[-1]))
    else:
        start = 0
        while start < len(idx_list):
            end = start + 1
            while end < len(idx_list) and min_len > idx_list[end] - idx_list[start]:
                end += 1
            if end < len(idx_list):
                res_list.append((idx_list[start], idx_list[end]))
            elif end == len(idx_list) and start < end - 1:
                res_list.append((idx_list[start], idx_list[end-1]))
            start = end

    return res_list


class SeqDataset(Dataset):
    def __init__(self, dataset: List[Dict], split_fn: Callable, min_len=10):
        assert isinstance(split_fn, FunctionType), "split_fn must be a function!"
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict('records')
        self._dataset = dataset
        self.idx_list = _get_span_list(split_fn(self._dataset), min_len=min_len)

    def __getitem__(self, item):
        start, end = self.idx_list[item]
        return {
            'data': self._dataset[start: end],
            'idx': item
        }

    def __len__(self):
        return len(self.idx_list)

    @property
    def raw_dataset(self):
        return self._dataset
