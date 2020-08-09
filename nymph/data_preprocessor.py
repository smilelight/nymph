
from typing import Union, List

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightutils import check_file

from .vocab import Vocab
from .base import BasePreprocessor


class NoColumnException(Exception):
    def __init__(self, column_name: str, data_name: str):
        super(NoColumnException, self).__init__()
        self._column_name = column_name
        self._data_name = data_name

    def __str__(self):
        return "no column named {} in {} dataset".format(self._column_name, self._data_name)


class NotFittedException(Exception):
    def __str__(self):
        return "the DataPreprocessor has not been fitted, please fit the data first!"


class DataPreprocessor(BasePreprocessor):
    def __init__(self, window_size=0):
        super(DataPreprocessor, self).__init__()
        self.vocabs = dict()
        self.num_scaler = StandardScaler()
        self.num_label_list = None
        self.target_vocab = None
        self._init_token = '<sos>'
        self._eos_token = '<eos>'
        self._init_num = 0.0
        self._eos_num = 1.0
        self.window_size = window_size
        self.feature_columns: List = []

    def set_target(self, target: Union[List, pd.Series]):
        vocab = Vocab(unk_token=None)
        vocab.add_words(target)
        vocab.build_vocab()
        self.target_vocab = vocab

    def fit(self, dataset: pd.DataFrame):
        if not self.feature_columns:
            self.feature_columns = dataset.columns.tolist()

        # 针对离散类型数据建立词表并使用Embedding转化为稠密向量
        for index in dataset:
            if dataset[index].dtype == 'object' or dataset[index].dtype == 'bool':
                vocab = Vocab(init_token=self._init_token, eos_token=self._eos_token)
                vocab.add_words(dataset[index].tolist())
                vocab.build_vocab()
                vocab.init_vectors()
                self.vocabs[index] = vocab
                print(index)

        # 针对连续性数据进行标准化
        def numeric_filter(dataset):
            return dataset.dtypes[(dataset.dtypes == 'float64') | (dataset.dtypes == 'int64')].index

        numeric_data = dataset[numeric_filter(dataset)]
        self.num_scaler.fit(numeric_data)
        self.num_label_list = numeric_filter(dataset).tolist()

    def fit_from_csv(self, csv_path: str, feature_columns: List, target_column: str):
        check_file(csv_path, 'csv')
        self.feature_columns = feature_columns
        data = pd.read_csv(csv_path)
        feature_data = data[feature_columns]

        self.fit(feature_data)

        if target_column not in data:
            raise NoColumnException(target_column, csv_path)
        else:
            self.set_target(data[target_column].tolist())

    def _check_index(self, indexs):
        for vocab in self.vocabs:
            if vocab not in indexs:
                raise IndexError
        for num_index in self.num_label_list:
            if num_index not in indexs:
                raise IndexError

    def _check_state(self):
        if not self.vocabs:
            raise NotFittedException

    def transform_feature(self, dataset: pd.DataFrame):
        self._check_state()
        self._check_index(list(dataset))
        tensor_list = []
        for index in dataset:
            if index in self.vocabs:
                ios_data = np.array([self.vocabs[index].init_token] * self.window_size).reshape((-1, 1))
                eos_data = np.array([self.vocabs[index].eos_token] * self.window_size).reshape((-1, 1))
                index_data = np.concatenate([ios_data, dataset[index].values.reshape((-1, 1)), eos_data])
                index_num = self.vocabs[index].numericalize(index_data)
                index_vec = self.vocabs[index].vectorize(torch.LongTensor(index_num))
                index_vec = torch.squeeze(index_vec, dim=1)
                tensor_list.append(index_vec)
        stand_num_vec = self.num_scaler.transform(dataset[self.num_label_list])
        num_size = stand_num_vec.shape[-1]
        ios_num = np.array([self._init_num]*self.window_size*num_size).reshape((self.window_size, num_size))
        eos_num = np.array([self._eos_num]*self.window_size*num_size).reshape((self.window_size, num_size))
        stand_num_vec = np.concatenate([ios_num, stand_num_vec, eos_num])
        tensor_list.append(torch.Tensor(stand_num_vec))
        total_data = torch.cat(tensor_list, dim=1).data
        res_data = torch.stack([total_data[i: i+self.window_size*2+1] for i in range(len(total_data) - self.window_size*2)])
        return res_data.data.numpy()

    def transform_target(self, target):
        self._check_state()
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        return self.target_vocab.numericalize(target.reshape((-1, 1)))

    def reverse_target(self, target_nums):
        self._check_state()
        return self.target_vocab.reverse(target_nums)

    def transform_raw(self, raw_data):
        self._check_state()
        df = pd.DataFrame(raw_data, columns=self.feature_columns)
        return self.transform_feature(df)

    @property
    def class_nums(self):
        if self.target_vocab is None:
            return 0
        else:
            return len(self.target_vocab)
