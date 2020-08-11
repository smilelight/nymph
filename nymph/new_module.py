
from typing import Union, List

import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from lightutils import logger

from .config import CONFIG
from .model import Config, BiLstmCrfClassifier
from .metric import get_score
from .data_preprocessor import DataPreprocessor
from .tool import EarlyStopping, save_dict_to_csv
from .dataset import ClsDataset

seed = 2020
torch.manual_seed(seed)


class SeqClassifier:
    def __init__(self):
        self._config = None
        self._model = None
        self.preprocessor = None

    def init_preprocessor(self, window_size=1):
        self.preprocessor = DataPreprocessor(window_size=window_size)
        return self.preprocessor

    def train(self, train_set: ClsDataset, dev_set=None, save_path=CONFIG['save_path']):
        class_num = self.preprocessor.class_nums
        batch_size = 256
        train_iter = DataLoader(train_set, batch_size=batch_size)
        item_shape = train_set[0][0].shape
        window_size = item_shape[0]
        feature_dimension = item_shape[1]
        self._config = Config(save_path=save_path, class_num=class_num, window_size=window_size,
                              feature_dimension=feature_dimension, batch_size=batch_size)
        bilstm_crf_classifier = BiLstmCrfClassifier(self._config)
        self._model = bilstm_crf_classifier
        opt = torch.optim.Adam(bilstm_crf_classifier.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, 25)

        early_stopping = EarlyStopping(score_mode=dev_set is not None)
        for epoch in range(200):
            self._model.train()
            acc_loss = 0
            for x_i, y_i in tqdm(train_iter):
                opt.zero_grad()
                x_i = x_i.permute(1, 0, 2)
                y_i = y_i.permute(1, 0)
                item_loss = (- self._model.loss(x_i, y_i)) / x_i.shape[0]
                acc_loss += item_loss.item()
                item_loss.backward()
                opt.step()
            scheduler.step()
            logger.info("learning rate is {}".format(opt.param_groups[0]["lr"]))
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_set:
                pass
                # dev_score = self.score(dev_set)
                # logger.info("dev score: {}".format(dev_score))
                # if early_stopping(new_record=dev_score):
                #     logger.info("reach early stopping patience, break training.")
                #     break
            else:
                if early_stopping(new_record=acc_loss):
                    logger.info("reach early stopping patience, break training.")
                    break

        self._config.save()
        self._model.save()
        self.preprocessor.save(path=save_path)

    def score(self, test_set: ClsDataset):
        self._model.eval()
        score_list = []
        test_iter = DataLoader(test_set, batch_size=256)
        for x_i, y_i in tqdm(test_iter):
            item_score = get_score(self._model, x_i, y_i.reshape((-1)))
            score_list.append(item_score)
        res = sum(score_list) / len(score_list)
        return res

    def summary(self, dataset: pd.DataFrame, output_path: str = 'summary.csv'):
        features = dataset[self.preprocessor.feature_columns]
        x = self.preprocessor.transform_feature(features)
        pred = self.predict(x)
        dataset['pred_label'] = [x[0] for x in pred]
        dataset['pred_prob'] = [x[1] for x in pred]
        # print(dataset[dataset['label'] == dataset['pred_label']])
        dataset.to_csv(output_path, index=False, encoding='utf8')
        logger.info("预测结果已经写入到{}文件中".format(output_path))

    def report(self, dataset: ClsDataset, output_path: str = 'report.csv', verbose=False):
        assert output_path.endswith('.csv'), 'output_path must endswith csv'
        pred = self.predict(dataset.x)
        pred_labels = [x[0] for x in pred]
        true_labels = dataset.y_raw.values.tolist()
        all_labels = self.preprocessor.target_vocab.itos
        if output_path:
            cls_rep = classification_report(true_labels, pred_labels, labels=all_labels, zero_division=1,
                                            output_dict=True)
            save_dict_to_csv(cls_rep, output_path)
            logger.info("已经成功将结果写入至{}中".format(output_path))
        if verbose:
            print(classification_report(true_labels, pred_labels, labels=all_labels, zero_division=1))

    def load(self, save_path=CONFIG['save_path']):
        config = Config.load(save_path)
        model = BiLstmCrfClassifier(config)
        model.load()
        self._config = config
        self._model = model
        self.preprocessor = DataPreprocessor.load(path=save_path)

    def predict(self, x: Union[List, pd.DataFrame]):
        assert isinstance(self.preprocessor, DataPreprocessor)
        if isinstance(x, list):
            x = self.preprocessor.transform_raw(x)
        elif isinstance(x, pd.DataFrame):
            x = self.preprocessor.transform_feature(x)
        self._model.eval()
        pred = self._model(torch.Tensor(x))
        soft_pred = torch.softmax(pred, dim=1)
        pred_prob, pred_idx = torch.max(soft_pred.cpu().data, dim=1)
        pred_cls = self.preprocessor.reverse_target(pred_idx.tolist())
        pred_prob = map(lambda n: round(n, 4), pred_prob.data.tolist())
        return list(zip(pred_cls, pred_prob))
