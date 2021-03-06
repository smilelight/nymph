# -*- coding: utf-8 -*-


class EarlyStopping:
    def __init__(self, patience=7, score_mode=True):
        self.patience = patience
        self.no_update_counter = 0
        self.best_record = 10000
        self.score_mode = score_mode
        if self.score_mode:
            self.best_record = -10000

    def update(self, new_record):
        self.no_update_counter += 1
        if self.score_mode:
            if new_record > self.best_record:
                self.best_record = new_record
                self.no_update_counter = 0
                return True
        else:
            if new_record < self.best_record:
                self.best_record = new_record
                self.no_update_counter = 0
                return True
        return False

    def stop(self):
        return self.no_update_counter >= self.patience

    @property
    def best_score(self):
        return self.best_record
