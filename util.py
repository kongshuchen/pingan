#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 下午4:31
# @Author  : meikun
# @Site    : 
# @File    : util.py
# @Software: PyCharm Community Edition

import numpy as np
### Gini

def ginic(actual, pred):
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n


def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:, 1]
    return ginic(a, p) / ginic(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def gini_score(preds, y):
    return gini_normalized(y, preds)