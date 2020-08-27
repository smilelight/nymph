# -*- coding: utf-8 -*-
import os

import pandas as pd
from nymph.data import NormDataset, split_dataset
from nymph.modules import NormClassifier

project_path = os.path.abspath(os.path.join(__file__, '../../'))
data_path = os.path.join(project_path, r'data\test.csv')
save_path = 'demo_saves'

if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv(data_path)
    # 构建分类器
    classifier = NormClassifier()

    # 加载分类器
    classifier.load(save_path)

    # 构建数据集
    norm_ds = NormDataset(data)

    # 预测模型
    pred = classifier.predict(norm_ds)
    print(pred)

    # 获取各类别分类结果，并保存信息至文件中
    classifier.report(norm_ds, 'report.csv')

    # 对数据进行预测，并将数据和预测结果写入到新的文件中
    classifier.summary(norm_ds, 'summary.csv')

