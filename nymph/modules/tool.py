# -*- coding: utf-8 -*-
import pandas as pd


def save_dict_to_csv(dict_data: dict, csv_path: str):
    indexes = list(dict_data.keys())
    columns = list(list(dict_data.values())[0].keys())
    data = []
    for row in dict_data:
        data.append([item for item in dict_data[row].values()])
    pd_data = pd.DataFrame(data, index=indexes, columns=columns)
    pd_data.to_csv(csv_path, encoding='utf8')
    return pd_data
