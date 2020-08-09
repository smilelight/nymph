
from collections import defaultdict, Counter, OrderedDict
from typing import Union, List, Set

import numpy as np
import torch
import torch.nn as nn

seed = 2020
torch.manual_seed(seed)


def _default_unk_index():
    return 0


class Vocab(object):
    def __init__(self, init_token=None, eos_token=None, pad_token="<pad>", unk_token="<unk>"):
        self.word_count = Counter()
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.itos = list()
        self.stoi = defaultdict()
        self._vectors = None

    def build_vocab(self, min_freq=1, max_size=None, specials_first=True):

        specials = list(OrderedDict.fromkeys(tok for tok in [self.unk_token, self.pad_token, self.init_token,
                                                             self.eos_token] if tok is not None))

        if specials_first:
            self.itos = list(specials)

        words_and_frequencies = sorted(self.word_count.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        max_size = None if max_size is None else max_size + len(self.itos)
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if not specials_first:
            self.itos.extend(list(specials))

        if '<unk>' in specials:
            self.stoi = defaultdict(_default_unk_index)

        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        return self

    def add_words(self, words: Union[List, Set]):
        self.word_count.update(words)
        return self

    def add_word(self, word):
        self.word_count[word] += 1
        return self

    def clear(self):
        self.word_count = Counter()
        self.itos = list()
        self.stoi = defaultdict()

    def __len__(self):
        return len(self.itos)

    def __contains__(self, item):
        return item in self.stoi

    def has_word(self, w):
        return self.__contains__(w)

    def __getitem__(self, w):
        return self.stoi[w]

    def to_index(self, w):
        return self.__getitem__(w)

    def to_word(self, idx):
        if isinstance(idx, int) and 0 <= idx < len(self.itos):
            return self.itos[idx]
        return None

    def __iter__(self):
        for word, idx in self.stoi.items():
            yield word, idx

    def numericalize(self, arr):
        assert len(set(len(row) for row in arr)) == 1, "每一行的长度都要相等！"
        return np.array([[self[x] for x in row] for row in arr])

    def init_vectors(self, dim=3):
        self._vectors = nn.Embedding(len(self), dim)

    def vectorize(self, arr):
        return self._vectors(arr).data

    def reverse(self, arr):
        return [self.to_word(x) for x in arr]
