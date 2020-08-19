# -*- coding: utf-8 -*-
import torch


class Normalizer:
    def __init__(self):
        self.std = torch.tensor(1)
        self.mean = torch.tensor(0)

    def fit(self, num_list):
        num_list = torch.tensor(num_list)
        self.std = torch.std(num_list)
        self.mean = torch.mean(num_list)

    def transform(self, num_list):
        return (num_list - self.mean) / self.std


if __name__ == '__main__':
    nums = torch.rand((2, 3))
    print(nums)

    nor = Normalizer()

    nor.fit(nums)

    print(nor.std, nor.mean)

    others = torch.rand((3, 4))

    print(nor.transform(others))
