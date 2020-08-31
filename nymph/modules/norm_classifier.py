
from typing import Union, List, Dict

import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from lightutils import logger

from ..core.config import CONFIG, DEVICE
from ..models.linear import Config, LinearClassifier
from ..data.dataset import NormDataset
from ..data.processor import NormProcessor
from ..process.metric import get_score
from ..process.train import EarlyStopping
from .tool import save_dict_to_csv

seed = 2020
torch.manual_seed(seed)


class NormClassifier:
    def __init__(self):
        self._config = None
        self._model = None
        self.data_processor = None
        self.batch_size = 32

    def init_data_processor(self, dataset:  Union[List[Dict], pd.DataFrame], target_name: str, un_fit_names=None):
        if un_fit_names is None:
            un_fit_names = []
        self.data_processor = NormProcessor()
        self.data_processor.fit(dataset, target_name, un_fit_names)
        return self.data_processor

    def train(self, train_set: Union[NormDataset, ConcatDataset], dev_set=None, save_path=CONFIG['save_path'], 
              log_dir: str = None):
        
        log_writer = SummaryWriter(log_dir) if log_dir else None
        if log_dir:
            logger.info("training logs will be written in {}".format(log_dir))
        batch_size = self.batch_size
        class_num = self.data_processor.target_nums
        feature_dimension = self.data_processor.feature_dimension

        train_iter = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                                collate_fn=self.data_processor.transform)
        self._config = Config(save_path=save_path, class_num=class_num,
                              feature_dimension=feature_dimension, batch_size=batch_size)
        self._model = LinearClassifier(self._config)
        print("model structure:", self._model)
        opt = torch.optim.Adam(self._model.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 20)

        early_stopping = EarlyStopping(score_mode=dev_set is not None, patience=30)
        for epoch in range(100):
            self._model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                opt.zero_grad()
                x = item['features']
                y = item['targets']
                pred = self._model(x)
                item_loss = F.cross_entropy(pred, y.reshape((-1)).to(DEVICE))
                acc_loss += item_loss.item()
                item_loss.backward()
                opt.step()
            scheduler.step()
            logger.info("learning rate is {}".format(opt.param_groups[0]["lr"]))
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if log_dir:
                log_writer.add_scalar('Loss/train', acc_loss, epoch)
            if dev_set:
                pass
                dev_score = self.score(dev_set)
                logger.info("dev score: {}".format(dev_score))
                if log_dir:
                    log_writer.add_scalar('F1/train', dev_score, epoch)
                if early_stopping.update(new_record=dev_score):
                    self._model.save_best_model()
                if early_stopping.stop():
                    logger.info("reach early stopping patience, break training. ")
                    break
            else:
                if early_stopping.update(new_record=acc_loss):
                    self._model.save_best_model()
                if early_stopping.stop():
                    logger.info("reach early stopping patience, break training.")
                    break

        logger.info("best record: {}".format(early_stopping.best_score))
        self._config.save()
        self._model.save()
        self.data_processor.save(path=save_path)

    def score(self, test_set: NormDataset):
        batch_size = self.batch_size
        self._model.eval()
        score_list = []
        test_iter = DataLoader(test_set, batch_size=batch_size, drop_last=False, collate_fn=self.data_processor.transform)
        for item in tqdm(test_iter):
            x = item['features']
            y = item['targets']
            item_score_list = get_score(self._model, x, y.reshape((-1)).to(DEVICE))
            score_list.append(item_score_list)
        res = sum(score_list) / len(score_list)
        return res

    def summary(self, dataset: NormDataset, output_path: str = 'summary.csv'):
        pred = self.predict(dataset)
        data_pd = pd.DataFrame(dataset.raw_dataset)
        data_pd['pred_label'] = [x for x in pred]
        data_pd.to_csv(output_path, index=False, encoding='utf8')
        logger.info("预测结果已经写入到{}文件中".format(output_path))

    def report(self, dataset: NormDataset, output_path: str = 'report.csv', verbose=False):
        assert output_path.endswith('.csv'), 'output_path must endswith csv'
        pred = self.predict(dataset)
        all_labels = self.data_processor.target_vocab
        target_name = self.data_processor.target_name
        true_labels = [x[target_name] for x in dataset.raw_dataset]
        if output_path:
            cls_rep = classification_report(true_labels, pred, labels=all_labels, zero_division=1,
                                            output_dict=True)
            save_dict_to_csv(cls_rep, output_path)
            logger.info("已经成功将结果写入至{}中".format(output_path))
        if verbose:
            print(classification_report(true_labels, pred, labels=all_labels, zero_division=1))

    def load(self, save_path=CONFIG['save_path']):
        config = Config.load(save_path)
        model = LinearClassifier(config)
        model.load()
        self._config = config
        self._model = model
        self.data_processor = NormProcessor.load(path=save_path)

    def predict(self, pred_set: NormDataset):
        batch_size = self.batch_size
        self._model.eval()
        pred_list = []
        pred_iter = DataLoader(pred_set, batch_size=batch_size, drop_last=False,
                               collate_fn=self.data_processor.transform)
        for item in tqdm(pred_iter):
            x = item['features']
            pred_item = self._model(x)
            soft_pred = torch.softmax(pred_item, dim=1)
            pred_prob, pred_idx = torch.max(soft_pred.cpu().data, dim=1)
            pred_list.append(pred_idx.tolist())
        labels_list = self.data_processor.reverse_target(pred_list)
        res_list = []
        for labels in labels_list:
            res_list.extend(labels)
        return res_list
