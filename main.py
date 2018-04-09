import numpy as np
import pandas as pd
import xgboost as xgb

from feature import get_time, transform_df, month_st, weekday_st, callstate_st, \
    id_size, longitude_st, latitude_st, height_st

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


def get_feature(df):
    df = transform_df(df)
    group_list = df.groupby('TERMINALNO')
    data_list = list()

    for key, group in group_list:
        size = id_size(group)
        long_st = longitude_st(group)
        lati_st = latitude_st(group)
        hei_st = height_st(group)
        month = month_st(group)
        weekday = weekday_st(group)
        callstate = callstate_st(group)
        data_list.append(pd.concat([long_st, lati_st, hei_st, size,
                                    month, weekday, callstate], axis=1))

    res = pd.concat(data_list).reset_index(drop=True)
    res = res.fillna(0)
    return res


def get_label(df):
    return df.groupby('TERMINALNO')['Y'].agg(np.mean).reset_index(drop=True)


def process():
    data = read_csv()
    data = get_time(data)
    train_feature = get_feature(data)
    train_label = get_label(data)

    params = {'eta': 0.025, 'max_depth': 4,
              'subsample': 0.9, 'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'min_child_weight': 100,
              'alpha': 4,
              'objective': 'reg:linear', 'eval_metric': 'auc', 'seed': 99, 'silent': False}
    model = xgb.train(params, xgb.DMatrix(train_feature, train_label), 1000)

    test_data = pd.read_csv(path_test)
    test_data.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                         "CALLSTATE"]
    test_data = get_time(test_data)
    test_feature = get_feature(test_data)

    res = pd.DataFrame(model.predict(xgb.DMatrix(test_feature), ntree_limit=model.best_ntree_limit))
    res.index.name='Id'
    res.columns=['Pred']
    res.to_csv('model/pred.csv')

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()