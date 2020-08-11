import pandas as pd


class EarlyStopping:
    def __init__(self, patience=7, score_mode=True):
        self.patience = patience
        self.no_update_counter = 0
        self.best_record = 10000
        self.score_mode = score_mode
        if self.score_mode:
            self.best_record = -10000

    def __call__(self, new_record):
        self.no_update_counter += 1
        if self.score_mode:
            if new_record > self.best_record:
                self.best_record = new_record
                self.no_update_counter = 0
        else:
            if new_record < self.best_record:
                self.best_record = new_record
                self.no_update_counter = 0

        return self.no_update_counter >= self.patience


def save_dict_to_csv(dict_data: dict, csv_path: str):
    indexes = list(dict_data.keys())
    columns = list(list(dict_data.values())[0].keys())
    data = []
    for row in dict_data:
        data.append([item for item in dict_data[row].values()])
    pd_data = pd.DataFrame(data, index=indexes, columns=columns)
    pd_data.to_csv(csv_path, encoding='utf8')
    return pd_data
