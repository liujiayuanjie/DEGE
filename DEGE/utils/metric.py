import numpy as np
from numpy import ndarray
from sklearn import metrics

def cal_auc(y_true: ndarray, y_pred: ndarray) -> float:
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    return auc

def cal_acc(y_true: ndarray, y_pred: ndarray) -> float:
    y_pred = np.around(y_pred)
    acc = (y_true == y_pred).sum() / y_true.shape[0]
    return acc