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
    classifier.init_data_processor(data, target_name='label')

    # 构建数据集
    norm_ds = SeqDataset(data, split_fn=split_fn, min_len=4)

    train_ratio = 0.7
    dev_ratio = 0.2
    test_ratio = 0.1

    train_ds, dev_ds, test_ds = split_dataset(norm_ds, (train_ratio, dev_ratio, test_ratio))

    # 训练模型
    # classifier.train(train_set=train_ds, dev_set=dev_ds, save_path=save_path)
    classifier.train(train_set=norm_ds, dev_set=norm_ds, save_path=save_path)

    # 测试模型
    test_score = classifier.score(norm_ds)
    print('test_score', test_score)

    # 预测模型
    pred = classifier.predict(norm_ds)
    print(pred)

