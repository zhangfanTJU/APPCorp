import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100

    return ' '.join([reformat(p, 2, True), reformat(r, 2, True), reformat(f1, 2, True)]), reformat(f1)


def reformat(num, n=2, return_str=False):
    res = float(format(num, '0.' + str(n) + 'f'))
    if return_str:
        return str(res)
    else:
        return res
