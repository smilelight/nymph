# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence


def get_mask(seq_lens, batch_first=False):
    masks = []
    for seq_len in seq_lens:
        mask_item = torch.ones(seq_len, dtype=torch.bool)
        masks.append(mask_item)
    return pad_sequence(masks, batch_first=batch_first)
