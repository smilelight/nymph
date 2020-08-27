# -*- coding: utf-8 -*-
import random

import torch
import numpy as np


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
