# -*- coding: utf-8 -*-
import os

import pandas as pd
from nymph.data import SeqDataset, split_dataset
from nymph.modules import SeqClassifier

project_path = os.path.abspath(os.path.join(__file__, '../../'))
data_path = os.path.join(project_path, r'data\test.csv')
save_path = 'demo_saves_seq'


def split_fn(dataset: list):
    return list(range(len(dataset)+1))


if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv(data_path)
    # 构建分类器
    classifier = SeqClassifier()

    # 加载分类器
    classifier.load(save_path)

    # 构建数据集
    seq_ds = SeqDataset(data, split_fn=split_fn, min_len=4)

    # 预测模型
    pred = classifier.predict(seq_ds)
    print(pred)

    # 获取各类别分类结果，并保存信息至文件中
    classifier.report(seq_ds, 'seq_demo_report.csv')

    # 对数据进行预测，并将数据和预测结果写入到新的文件中
    classifier.summary(seq_ds, 'seq_demo_summary.csv')

