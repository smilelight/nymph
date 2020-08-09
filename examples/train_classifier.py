import os

import pandas as pd
from sklearn.model_selection import train_test_split
from nymph.module import Classifier
from nymph.dataset import ClsDataset

project_path = os.path.abspath(os.path.join(__file__, '../../'))
data_path = os.path.join(project_path, r'data\test.csv')
columns = ['fuck', 'shit']
target = 'label'

if __name__ == '__main__':
    # 读取原始数据
    data = pd.read_csv(data_path)

    # 构建分类器并初始化预处理器
    classifier = Classifier()
    data_preprocessor = classifier.init_preprocessor(window_size=0)
    data_preprocessor.fit(data[columns])
    data_preprocessor.set_target(data[target])

    # 划分并构建训练测试数据集
    x_train, x_test, y_train, y_test = train_test_split(data[columns], data[target], test_size=0.1, random_state=2020)

    train_set = ClsDataset(data_preprocessor, x_train, y_train)
    test_set = ClsDataset(data_preprocessor, x_test, y_test)

    # 训练模型
    classifier.train(train_set=train_set, dev_set=test_set)

    # 评估模型分数
    test_score = classifier.score(test_set=test_set)
    print(test_score)

    # 获取各类别分类结果，并保存信息至文件中
    print(classifier.report(test_set, output_path='report.txt'))

    # 对数据进行预测，并将数据和预测结果写入到新的文件中
    classifier.summary(data, 'summary.csv')
