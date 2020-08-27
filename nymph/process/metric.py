import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def get_score(model, x, y, score_type='f1'):
    metrics_map = {
        'f1': f1_score,
        'p': precision_score,
        'r': recall_score,
        'acc': accuracy_score
    }
    metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
    assert len(x) == len(y)
    vec_predict = model(x)
    soft_predict = torch.softmax(vec_predict, dim=1)
    predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
    y = y.view(-1).cpu().data.numpy()
    return metric_func(predict_index, y, average='micro')


def get_seq_score(model, x, seq_lens, y):
    metric_func = f1_score
    score_list = []
    pred = model(x, seq_lens)
    for y_item, pred_item in zip(y.T, pred):
        score_list.append(metric_func(y_item[:len(pred_item)], pred_item, average='micro'))
    return score_list
