
import torch
from lightutils import logger
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info('using device: {}'.format(DEVICE))
DEFAULT_CONFIG = {
    'save_path': './saves'
}

CONFIG = {
    'save_path': './saves'
}
