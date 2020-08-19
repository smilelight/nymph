# -*- coding: utf-8 -*-
from typing import List
from abc import abstractmethod

import numpy as np

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
        self.vectors = vectors

    def fit(self, seq_list: List[str]):
        for sentence in seq_list:
            for token in self.tokenizer(sentence):
                self.vocab.add_word(token)
        if not self.vectors:
            self.vocab.init_vectors()

    def numericalize(self, arr):
        assert len(set(len(row) for row in arr)) == 1, "每一行的长度都要相等！"
        return np.array([[self.vocab[x] for x in row] for row in arr])

    def vectorize(self, arr):
        return self.vocab.vectors(arr).data


class WordField(Field):
    def __init__(self, vectors=None):
        super().__init__()
        self.use_vocab = True
        self.vocab = Vocab()
        self.vectors = vectors

    def fit(self, word_list: List[str]):
        self.vocab.add_words(word_list)
        if not self.vectors:
            self.vocab.init_vectors()

    def numericalize(self, arr):
        assert len(set(len(row) for row in arr)) == 1, "每一行的长度都要相等！"
        return np.array([[self.vocab[x] for x in row] for row in arr])

    def vectorize(self, arr):
        return self.vocab.vectors(arr).data


class NumField(Field):
    def __init__(self):
        super().__init__()
        self._normalizer = Normalizer()

    def fit(self, num_list):
        self._normalizer.fit(num_list)

    def numericalize(self, arr):
        return arr

    def vectorize(self, arr):
        return arr
