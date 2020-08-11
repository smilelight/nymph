import pandas as pd
from sklearn.model_selection import train_test_split
# from nymph.module import Classifier
from nymph.new_module import SeqClassifier
from nymph.dataset import ClsDataset

data_path = r"D:\data\Projects\训练数据\冀北风电调峰控制功能说明-rec-labeled-train.csv"

# 读取训练原始数据
columns = ['text_feature', 'is_center', 'is_bold', 'is_list_item', 'name', 'text_indent', 'size', 'font_size']
target = 'para_type'

if __name__ == '__main__':
    data = pd.read_csv(data_path)

    # 构建分类器并初始化预处理器
    classifier = SeqClassifier()
    data_preprocessor = classifier.init_preprocessor(window_size=1)
    data_preprocessor.fit(data[columns])
    data_preprocessor.set_target(data[target])

    # 划分并构建训练测试数据集
    x_train, x_test, y_train, y_test = train_test_split(data[columns], data[target], test_size=0.1, random_state=2020)

    train_set = ClsDataset(data_preprocessor, x_train, y_train, seq=True)
    test_set = ClsDataset(data_preprocessor, x_test, y_test, seq=True)

    x_0, y_0 = train_set[0]
    print(x_0)
    print(y_0)
    print(x_0.shape)
    print(y_0.shape)

    # 训练模型
    classifier.train(train_set=train_set, dev_set=test_set)
