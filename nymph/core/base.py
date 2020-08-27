
import os
import pickle

import torch
import torch.nn as nn
from lightutils import logger

from .config import DEFAULT_CONFIG, CONFIG


class BaseProcessor(object):
    base_name = 'data_processor.pkl'

    def __init__(self):
        pass

    @staticmethod
    def load(path=CONFIG['save_path']):
        preprocessor_path = os.path.join(path, BaseProcessor.base_name)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info('loading preprocessor from {}'.format(preprocessor_path))
        preprocessor.save_path = path
        return preprocessor

    def save(self, path=CONFIG['save_path']):
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, BaseProcessor.base_name)
        with open(os.path.join(path, BaseProcessor.base_name), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved processor to {}'.format(config_path))


class BaseConfig(object):
    base_name = 'config.pkl'

    def __init__(self):
        pass

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path']):
        config_path = os.path.join(path, BaseConfig.base_name)
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logger.info('loading config from {}'.format(config_path))
        config.save_path = path
        return config

    def save(self, path=None):
        if not hasattr(self, 'save_path'):
            raise AttributeError('config object must init save_path attr in init method!')
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, BaseConfig.base_name)
        with open(os.path.join(path, BaseConfig.base_name), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved config to {}'.format(config_path))


class BaseModel(nn.Module):
    base_name = 'model.pkl'
    best_model_name = 'best_model.pkl'

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, BaseModel.base_name)
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loading model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        best_model_path = os.path.join(path, BaseModel.best_model_name)
        model_path = os.path.join(path, BaseModel.base_name)
        if os.path.isfile(best_model_path):
            if os.path.isfile(model_path):
                os.remove(model_path)
            os.rename(best_model_path, model_path)
        else:
            torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))

    def save_best_model(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, BaseModel.best_model_name)
        torch.save(self.state_dict(), model_path)
        # logger.info('saved best model to {}'.format(model_path))
