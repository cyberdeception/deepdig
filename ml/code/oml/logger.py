import os
import logging
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_path):
        self.log_dir = log_path
        if not os.path.exists(os.path.join(self.log_dir, 'data')):
            os.makedirs(os.path.join(self.log_dir, 'data'))

        self.writer = SummaryWriter(os.path.join(log_path, 'data'))
        self.n_iter = 0

    def log_value(self, name, value):
        self.writer.add_scalar(os.path.join(self.log_dir, 'data', name), value, self.n_iter)
        return self

    def step(self):
        self.n_iter += 1


class OMLLogger:
    @staticmethod
    def getLogger(path):
        logging.basicConfig(
            filename=path,
            format='%(asctime)s--%(module)s--%(funcName)s--%(levelname)s %(message)s',
            level=logging.DEBUG
        )
        return logging.getLogger()
