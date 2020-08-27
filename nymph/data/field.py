# -*- coding: utf-8 -*-
from typing import List, Union
from abc import abstractmethod

import numpy as np
import torch

from .vocab import Vocab
from .tokenize import default_tokenizer
from .scale import Normalizer


class Field:
    def __init__(self):
        self.use_vocab = False

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def numericalize(self, *args, **kwargs):
        pass

    @abstractmethod
    def vectorize(self, *args, **kwargs):
        pass


class SeqField(Field):
    def __init__(self, vectors=None, tokenizer=default_tokenizer):
        super().__init__()
        self.use_vocab = True
        self.vocab = Vocab()
        self.tokenizer = tokenizer
        self._vectors = vectors

    def fit(self, seq_list: List[str]):
        for sentence in seq_list:
            for token in self.tokenizer(sentence):
                self.vocab.add_word(token)
        self.vocab.build_vocab()
        if not self._vectors:
            self.vocab.init_vectors()
        else:
            self.vocab.vectors = self._vectors

    def numericalize(self, arr: List[str]):
        tokens_list = [self.tokenizer(x) for x in arr]
        return np.array([[self.vocab[x] for x in row] for row in tokens_list])

    def vectorize(self, arr: Union[np.ndarray, torch.LongTensor]) -> torch.LongTensor:
        if not isinstance(arr, torch.LongTensor):
            arr = torch.LongTensor(arr)
        return self.vocab.vectors(arr).data

    @property
    def word2idx(self):
        return self.vocab.stoi

    @property
    def idx2word(self):
        return self.vocab.itos


class WordField(Field):
    def __init__(self, vectors=None, has_unk=True):
        super().__init__()
        self.use_vocab = True
        if has_unk:
            self.vocab = Vocab()
        else:
            self.vocab = Vocab(unk_token=None)
        self._vectors = vectors

    def fit(self, word_list: List[str]):
        self.vocab.add_words(word_list)
        self.vocab.build_vocab()
        if not self._vectors:
            self.vocab.init_vectors()
        else:
            self.vocab.vectors = self._vectors

    def numericalize(self, arr: List[str]):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if len(arr.shape) == 1:
            arr = arr.reshape((-1, 1))
        return np.array([[self.vocab[x] for x in row] for row in arr])

    def vectorize(self, arr: Union[np.ndarray, torch.LongTensor]) -> torch.LongTensor:
        if not isinstance(arr, torch.LongTensor):
            arr = torch.LongTensor(arr)
        return self.vocab.vectors(arr).data

    def reverse(self, num_list: list):
        return self.vocab.reverse(num_list)

    @property
    def word2idx(self):
        return self.vocab.stoi

    @property
    def idx2word(self):
        return self.vocab.itos

    @property
    def vec_dim(self):
        return self.vocab.vec_dim


class NumField(Field):
    def __init__(self):
        super().__init__()
        self._normalizer = Normalizer()

    def fit(self, num_list):
        if not isinstance(num_list, np.ndarray):
            num_list = np.array(num_list, dtype=np.float)
        self._normalizer.fit(num_list)

    def numericalize(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float)
        if len(arr.shape) == 1:
            arr = arr.reshape((-1, 1))
        return arr

    def vectorize(self, arr: Union[np.ndarray, torch.LongTensor]) -> torch.LongTensor:
        if not isinstance(arr, torch.LongTensor):
            arr = torch.LongTensor(arr)
        return self._normalizer.transform(arr)
