# -*- coding: utf-8 -*-
from typing import Dict, List, Union

import torch
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from .field import WordField, NumField
from ..core.base import BaseProcessor


class NormProcessor(BaseProcessor):
    def __init__(self):
        super(NormProcessor, self).__init__()
        self._field_dict = dict()
        self._un_fit_name_list: List[str] = []
        self._target_name = None

    def fit(self, dataset:  Union[List[Dict], pd.DataFrame], target_name: str, un_fit_names=None):
        if un_fit_names is None:
            un_fit_names = []
        self._un_fit_name_list = un_fit_names
        self._target_name = target_name
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict('records')
        assert len(dataset) > 0
        names = dataset[0].keys()
        assert target_name in names, "{}应该在name列表中".format(target_name)
        for name in names:
            if name in self._un_fit_name_list:
                continue
            if name == target_name:
                self.add_field(name, [item[name] for item in dataset], has_unk=False)
            else:
                self.add_field(name, [item[name] for item in dataset])

    def add_field(self, file_name: str, file_data: List, has_unk: bool = True):
        assert len(file_data) > 0
        if type(file_data[0]) in [str, bool]:
            filed = WordField(has_unk=has_unk)
            filed.fit(file_data)
            self._field_dict[file_name] = filed
        elif type(file_data[0]) in [int, float]:
            filed = NumField()
            filed.fit(file_data)
            self._field_dict[file_name] = filed
        else:
            raise TypeError

    @property
    def target_nums(self):
        return len(self._field_dict[self._target_name].idx2word)

    @property
    def target_name(self):
        return self._target_name

    @property
    def target_vocab(self):
        return self._field_dict[self._target_name].idx2word

    @property
    def feature_dimension(self):
        dimension = 0
        for field in self._field_dict:
            if field == self._target_name:
                continue
            if isinstance(self._field_dict[field], WordField):
                dimension += self._field_dict[field].vec_dim
            elif isinstance(self._field_dict[field], NumField):
                dimension += 1
        return dimension

    def numericalize(self, dataset: List[Dict]):
        assert len(dataset) > 0
        # print(dataset[0])
        names = dataset[0].keys()
        features = dict()
        target = None
        for name in names:
            if name in self._un_fit_name_list:
                continue
            assert name in self._field_dict
            num_item = self._field_dict[name].numericalize([item[name] for item in dataset])
            if name == self._target_name:
                target = num_item
            else:
                features[name] = num_item
        return features, target

    def vectorize(self, num_dict: dict):
        feature_vec = []
        for name in num_dict:
            if name == self._target_name:
                continue
            else:
                item_vec = self._field_dict[name].vectorize(num_dict[name])
                if len(item_vec.shape) == 3:
                    feature_vec.append(item_vec.squeeze(dim=1))
                else:
                    feature_vec.append(item_vec)
        res = torch.cat(feature_vec, dim=1)
        return res

    def transform(self, data: list):
        feature_nums, targets = self.numericalize(data)
        features = self.vectorize(feature_nums)
        if targets is not None:
            targets = torch.LongTensor(targets)
        return {
            'features': features,
            'targets': targets
        }

    def reverse_target(self, num_list: list):
        target_list = []
        for num_item in num_list:
            target_list.append(self._field_dict[self._target_name].reverse(num_item))
        return target_list


class SeqProcessor(NormProcessor):

    def transform(self, data: list):
        data.sort(key=lambda x: len(x['data']), reverse=True)
        features = []
        seq_lens = []
        targets = []
        indexes = []
        for item in data:
            item_data = item['data']
            item_idx = item['idx']
            indexes.append(item_idx)
            seq_lens.append(len(item_data))
            feature_nums, target = self.numericalize(item_data)
            feature_vec = self.vectorize(feature_nums)
            features.append(feature_vec)
            if target is not None:
                targets.append(target)
        if targets:
            targets = np.stack([np.concatenate((target.reshape(-1), np.zeros(max(seq_lens) - len(target), dtype=int)))
                                for target in targets]).T
        padded_seq = pad_sequence(features)
        return {
            'features': padded_seq,
            'seq_lens': seq_lens,
            'targets': targets,
            'indexes': indexes
        }
