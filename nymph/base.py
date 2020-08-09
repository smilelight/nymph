
import os
import pickle

import torch
import torch.nn as nn
from lightutils import logger

from .config import DEFAULT_CONFIG, CONFIG


class BasePreprocessor(object):
    def __init__(self):
        pass

    @staticmethod
    def load(path=CONFIG['save_path']):
        preprocessor_path = os.path.join(path, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info('loading preprocessor from {}'.format(preprocessor_path))
        preprocessor.save_path = path
        return preprocessor

    def save(self, path=CONFIG['save_path']):
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, 'preprocessor.pkl')
        with open(os.path.join(path, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved preprocessor to {}'.format(config_path))


class BaseConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path']):
        config_path = os.path.join(path, 'config.pkl')
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
        config_path = os.path.join(path, 'config.pkl')
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved config to {}'.format(config_path))


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loading model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))
