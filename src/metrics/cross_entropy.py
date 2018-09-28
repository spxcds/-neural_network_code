#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : cross_entropy.py
# @Time    : 2018/09/28
# @Author  : spxcds (spxcds@gmail.com)

from math import log
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer


class CrossEntropy(object):
    eps = 1e-15

    @staticmethod
    def calculate(y_true, y_pred, labels=None):
        lb = LabelBinarizer()
        if labels is not None:
            lb.fit(labels)
        else:
            lb.fit(y_true)

        y_true = lb.transform(y_true)
        y_pred = np.clip(y_pred, CrossEntropy.eps, 1 - CrossEntropy.eps)

        loss = 0.0
        for label, prob in zip(y_true, y_pred):
            for i in range(len(label)):
                loss += -label[i] * log(prob[i])

        loss /= len(y_true)
        return loss


if __name__ == '__main__':
    y_true = [0, 0, 2, 1, 2]
    y_pred = [[0.9, 0.05, 0.05], [0.88, 0.02, 0.1], [0, 1, 0], [0.03, 0.07, 0.9], [0.1, 0.8, 0.1]]

    print(CrossEntropy.calculate(y_true=y_true, y_pred=y_pred))
    print(log_loss(y_true=y_true, y_pred=y_pred))
