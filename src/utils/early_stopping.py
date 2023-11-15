import logging

import numpy as np


class EarlyStopping:
    def __init__(self, name, min_delta=0, patience=0, mode='min', baseline=None):
        self.name = name
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.mode = mode
        self.baseline = baseline
        self.best_model_params = None
        self.best_metrics = None

        if mode not in ['min', 'max']:
            logging.info('EarlyStopping mode %s is unknown, fallback to min mode.', mode)
            mode = 'min'

        if mode == 'min':
            self.compare_op = np.less_equal
        else:
            self.compare_op = np.greater_equal

        if self.mode == 'min':
            self.min_delta *= -1

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.mode == 'min' else -np.Inf

    def should_stop(self, value, model_params=None, metrics=None):
        if self.compare_op(value - self.min_delta, self.best):
            self.best = value
            self.best_model_params = model_params
            self.best_metrics = metrics
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def get_best_model_params(self):
        return self.best_model_params

    def get_best_metrics(self):
        return self.best_metrics
