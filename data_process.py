import pandas as pd
import numpy as np
import os
import shutil
import gc

def extract_feature(path, dtypes, save_path, data_process_params=None, target=None):
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    # os.makedirs(save_path)
    #
    # os.chdir(save_path)
    # features_dir = os.path.join(save_path, 'datas/')
    # os.makedirs('datas/')

    process_params = {}
    df = pd.read_csv(path, dtype=dtypes)

    # (1) 过滤掉方向未知或速度未知的记录, 刪掉 69306-69191 = 115 個數據
    df = df.loc[(df['DIRECTION'] >= 0) & (df['SPEED'] >= 0)].copy()

    # (3) get time feature
    df['hour'] = pd.to_datetime(df.TIME).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.TIME).dt.day.astype('uint8')
    df['wday'] = pd.to_datetime(df.TIME).dt.dayofweek.astype('uint8')
    df['month'] = pd.to_datetime(df.TIME).dt.month.astype('uint8')
    df['year'] = pd.to_datetime(df.TIME).dt.year.astype('uint8')

    del df['TIME']


    # 統計一個用戶hour大於23, 小於 8 的數量
    # df_group = df.groupby(['', 'hour'])

    # 統計一個用戶的TRIP_ID 數量
    df_group = df.groupby('TERMINALNO')['TRIP_ID'].max().reset_index().\
        rename(index=str, columns={'TRIP_ID': 'TRIP_ID_max'})

    # 統計一個用戶的LONGITUDE var
    gp = df.groupby('TERMINALNO')['LONGITUDE'].var().reset_index().\
        rename(index=str, columns={'LONGITUDE': 'LONGITUDE_var'})
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()

    # 統計一個用戶的LATITUDE var
    gp = df.groupby('TERMINALNO')['LATITUDE'].var().reset_index().rename(index=str, columns={'LATITUDE': 'LATITUDE_var'})
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()

    # 統計一個用戶的 DIRECTION var
    gp = df.groupby('TERMINALNO')['DIRECTION'].var().reset_index().rename(index=str, columns={'DIRECTION': 'DIRECTION_var'})
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()

    # 統計一個用戶的 HEIGHT 的 var
    gp = df.groupby('TERMINALNO')['HEIGHT'].var().reset_index().rename(index=str, columns={'HEIGHT': 'HEIGHT_var'})
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()

    # 統計一個用戶的 SPEED 的 var
    gp = df.groupby('TERMINALNO')['SPEED'].var().reset_index().rename(index=str, columns={'SPEED': 'SPEED_var'})
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()

    # # 統計一個用戶的 CALLSTATE 的 各個數字的count
    # gp = df.groupby(['TERMINALNO', 'CALLSTATE']).size()
    # # print(gp)
    # del gp
    # gc.collect()
    gp = df.groupby('TERMINALNO')['Y'].max().reset_index()
    df_group = df_group.merge(gp, on=['TERMINALNO'], how='left')
    del gp
    gc.collect()


    return df_group

