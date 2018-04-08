import os
import csv
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import recall_score,precision_score,r2_score,accuracy_score,classification_report,log_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from lightgbm.sklearn import LGBMClassifier

path_train = "/data/dm/train.csv"  # 训练文件
path_train = './data/dm/train.csv'
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    return tempdata
def time_change(num):
    num=int(float(num))
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(num))

def get_time(data):
    data['TIME']=pd.to_datetime(data['TIME'].apply(time_change))
    data['day']=[i.day for i in data['TIME']]
    data['month']=[i.month for i in data['TIME']]
    data['hour'] =[i.hour for i in data['TIME']]
    data['weekday']=[i.weekday() for i in data['TIME']]
    return data


def get_feature(data,dir_bins,speed_bins):
    feature = []
    person = data['TERMINALNO'].unique().tolist()
    print('person num',len(person))
    for i in person[:]:
        temp = data[data['TERMINALNO'] == i]
        num_count = len(temp)
        trip_num = len(temp['TRIP_ID'].unique())
        feature.append(trip_num)
        for month in range(1, 13):
            month_count = len(temp[temp['month'] == month]) / num_count
            # print(month,'month count',month_count)
            feature.append(month_count)
        for hour in range(0, 24):
            hour_count = len(temp[temp['hour'] == hour]) / num_count
            # print(hour,'hour count',hour_count)
            feature.append(hour_count)
        for weekday in range(0, 7):
            weekday_count = len(temp[temp['weekday'] == weekday]) / num_count
            # print(weekday,'week count',weekday_count)
            feature.append(weekday_count)
        for direction in range(len(dir_bins) - 1):
            # print(direction)
            direc_count = len(
                temp[(temp['DIRECTION'].astype(float) < dir_bins[direction + 1]) & (temp['DIRECTION'].astype(float) > dir_bins[direction])])
            # print(dir_bins[direction],dir_bins[direction+1],'dir count is ',direc_count/num_count)
            feature.append(direc_count / num_count)
        for speed in range(len(speed_bins) - 1):
            # print(speed)
            speed_count = len(temp[(temp['SPEED'].astype(float) < speed_bins[speed + 1]) & (temp['SPEED'].astype(float) > speed_bins[speed])])
            # print(speed_bins[speed],speed_bins[speed+1],'speed count is ',speed_count/num_count)
            feature.append(speed_count / num_count)
        for call in range(0, 5):
            call_num = len(temp[temp['CALLSTATE'] == call]) / num_count
            # print(call,'call_num',call_num)
            feature.append(call_num)
        # print(temp['LONGITUDE'].mean(),temp['LONGITUDE'].min(),temp['LONGITUDE'].max(),(temp['LONGITUDE'].max()-temp['LONGITUDE'].min()),
        #      temp['HEIGHT'].mean(),temp['HEIGHT'].min(),temp['HEIGHT'].max(),(temp['HEIGHT'].max()-temp['HEIGHT'].min()),
        #      temp['LATITUDE'].mean(),temp['LATITUDE'].min(),temp['LATITUDE'].max(),(temp['LATITUDE'].max()-temp['LATITUDE'].min()))
        feature.append(temp['LONGITUDE'].astype(float).mean())
        feature.append(temp['LONGITUDE'].astype(float).min())
        feature.append(temp['LONGITUDE'].astype(float).max())
        feature.append(temp['LONGITUDE'].astype(float).max() - temp['LONGITUDE'].astype(float).min())
        feature.append(temp['HEIGHT'].astype(float).mean())
        feature.append(temp['HEIGHT'].astype(float).min())
        feature.append(temp['HEIGHT'].astype(float).max())
        feature.append(temp['HEIGHT'].astype(float).max() - temp['HEIGHT'].astype(float).min())
        feature.append(temp['LATITUDE'].astype(float).mean())
        feature.append(temp['LATITUDE'].astype(float).min())
        feature.append(temp['LATITUDE'].astype(float).max())
        feature.append(temp['LATITUDE'].astype(float).max() - temp['LATITUDE'].astype(float).min())
        feature.append(temp['Y'].astype(float).mean())
    df=pd.DataFrame(np.array(feature).reshape(-1,79))
    print(df.shape)
    return df

def get_feature_test(data,dir_bins,speed_bins):
    feature = []
    person = data['TERMINALNO'].unique().tolist()
    print('person num',len(person))
    for i in person[:]:
        temp = data[data['TERMINALNO'] == i]
        num_count = len(temp)
        trip_num = len(temp['TRIP_ID'].unique())
        feature.append(trip_num)
        for month in range(1, 13):
            month_count = len(temp[temp['month'] == month]) / num_count
            # print(month,'month count',month_count)
            feature.append(month_count)
        for hour in range(0, 24):
            hour_count = len(temp[temp['hour'] == hour]) / num_count
            # print(hour,'hour count',hour_count)
            feature.append(hour_count)
        for weekday in range(0, 7):
            weekday_count = len(temp[temp['weekday'] == weekday]) / num_count
            # print(weekday,'week count',weekday_count)
            feature.append(weekday_count)
        for direction in range(len(dir_bins) - 1):
            # print(direction)
            direc_count = len(
                temp[(temp['DIRECTION'] < dir_bins[direction + 1]) & (temp['DIRECTION'] > dir_bins[direction])])
            # print(dir_bins[direction],dir_bins[direction+1],'dir count is ',direc_count/num_count)
            feature.append(direc_count / num_count)
        for speed in range(len(speed_bins) - 1):
            # print(speed)
            speed_count = len(temp[(temp['SPEED'] < speed_bins[speed + 1]) & (temp['SPEED'] > speed_bins[speed])])
            # print(speed_bins[speed],speed_bins[speed+1],'speed count is ',speed_count/num_count)
            feature.append(speed_count / num_count)
        for call in range(0, 5):
            call_num = len(temp[temp['CALLSTATE'] == call]) / num_count
            # print(call,'call_num',call_num)
            feature.append(call_num)
        # print(temp['LONGITUDE'].mean(),temp['LONGITUDE'].min(),temp['LONGITUDE'].max(),(temp['LONGITUDE'].max()-temp['LONGITUDE'].min()),
        #      temp['HEIGHT'].mean(),temp['HEIGHT'].min(),temp['HEIGHT'].max(),(temp['HEIGHT'].max()-temp['HEIGHT'].min()),
        #      temp['LATITUDE'].mean(),temp['LATITUDE'].min(),temp['LATITUDE'].max(),(temp['LATITUDE'].max()-temp['LATITUDE'].min()))
        feature.append(temp['LONGITUDE'].mean())
        feature.append(temp['LONGITUDE'].min())
        feature.append(temp['LONGITUDE'].max())
        feature.append(temp['LONGITUDE'].max() - temp['LONGITUDE'].min())
        feature.append(temp['HEIGHT'].mean())
        feature.append(temp['HEIGHT'].min())
        feature.append(temp['HEIGHT'].max())
        feature.append(temp['HEIGHT'].max() - temp['HEIGHT'].min())
        feature.append(temp['LATITUDE'].mean())
        feature.append(temp['LATITUDE'].min())
        feature.append(temp['LATITUDE'].max())
        feature.append(temp['LATITUDE'].max() - temp['LATITUDE'].min())

    df=pd.DataFrame(np.array(feature).reshape(-1,78))
    print('test_df',df.shape)
    df.index=person
    return df

def process():

    data=read_csv()
    print('train\n',data.head())
    data=get_time(data)

    dir_bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    speed_bins = [0, 30, 60, 90, 120, 150]
    df=get_feature(data, dir_bins, speed_bins)
    y = (df[78] > 0).astype(int)
    X = df.drop(78, axis=1)
    clf = LGBMClassifier().fit(X,y)
    testdata = pd.read_csv(path_test)
    testdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE"]
    print('test\n',testdata.head())
    data_test=testdata
    data_test = get_time(data_test)
    df_test=get_feature_test(data_test,dir_bins,speed_bins)
    res=pd.DataFrame(clf.predict_proba(df_test)).iloc[:,1:2]
    res.index.name='Id'
    res.columns=['Pred']
    print(res.head())
    res.to_csv('model/pred.csv')

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()