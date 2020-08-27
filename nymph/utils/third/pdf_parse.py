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


def pdf_split_fn(dataset: List[Dict]):
    idx_list = []
    for i, item in enumerate(dataset):
        if is_breakpoint(item):
            idx_list.append(i)
    if 0 not in idx_list:
        idx_list.insert(0, 0)
    if len(dataset) not in idx_list:
        idx_list.append(len(dataset))
    return idx_list


def pdf_label_parse(tags: List[str], ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]
