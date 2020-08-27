# -*- coding: utf-8 -*-

from typing import Dict, List


def is_breakpoint(item: dict):
    # if item['text_feature'] != 'text':
    if item['text_feature'] not in ['table', 'text', 'image']:
        return True
    if item['is_center'] is True:
        return True
    if item['is_bold'] is True:
        return True
    return False


def doc_split_fn(dataset: List[Dict]):
    idx_list = []
    for i, item in enumerate(dataset):
        if is_breakpoint(item):
            idx_list.append(i)
    if 0 not in idx_list:
        idx_list.insert(0, 0)
    if len(dataset) not in idx_list:
        idx_list.append(len(dataset))
    return idx_list


def doc_label_parse(labels: List[str]):
    return labels
