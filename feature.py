#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 上午11:22
# @Author  : meikun
# @Site    : 
# @File    : feature.py
# @Software: PyCharm Community Edition

import time
import pandas as pd
import numpy as np


def time_change(num):
    num=int(float(num))
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(num))


def get_time(data):
    data['TIME']=pd.to_datetime(data['TIME'].apply(time_change))
    data['day']=[i.day for i in data['TIME']]
    data['month']=[i.month for i in data['TIME']]
    data['hour']=[i.hour for i in data['TIME']]
    data['weekday']=[i.weekday() for i in data['TIME']]
    return data


def transform_df(df):
    one_hot_all = {c: list(df[c].unique()) for c in df.columns
                   if c not in ['Y'] and 20 > len(df[c].unique()) > 1}
    for (k, v) in one_hot_all.items():
        for val in v:
            df[k + '_oh_' + str(val)] = (df[k].values == val).astype(np.int)

    return df


def month_st(df):
    month_col = [col for col in list(df.columns.values) if col.startswith('month_')]
    return df.groupby('TERMINALNO')[month_col].agg(np.sum).reset_index(drop=True)


def weekday_st(df):
    weekday_col = [col for col in list(df.columns.values) if col.startswith('weekday_')]
    return df.groupby('TERMINALNO')[weekday_col].agg(np.sum).reset_index(drop=True)


def callstate_st(df):
    callstate_col = [col for col in list(df.columns.values) if col.startswith('callstate_')]
    return df.groupby('TERMINALNO')[callstate_col].agg(np.sum).reset_index(drop=True)


def id_size(df):
    return pd.DataFrame({'count': df.groupby('TERMINALNO').size()}).reset_index(drop=True)


def longitude_st(df):
    return df.groupby('TERMINALNO')['LONGITUDE'].agg(['sum',
           'max','min','mean','std']).add_prefix('LONGITUDE_').reset_index(drop=True)


def latitude_st(df):
    return df.groupby('TERMINALNO')['LATITUDE'].agg(['sum',
           'max','min','mean','std']).add_prefix('LATITUDE_').reset_index(drop=True)


def height_st(df):
    return df.groupby('TERMINALNO')['HEIGHT'].agg(['sum',
           'max','min','mean','std']).add_prefix('HEIGHT_').reset_index(drop=True)

