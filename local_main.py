import os
import csv
import pandas as pd
import time
import data_process
from sklearn.model_selection import train_test_split
from gbm_model import lgb_modelfit_nocv
import numpy as np

np.random.seed(2018)


path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

# path_train = '/home/ksc/PycharmProjects/pingan/data/train.csv'



CURRENT_PATH = os.getcwd()

train_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8',
                'Y': 'float32'}

test_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8'}

def read_csv():
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    print('>>>[1].Preprocessing train data and test data...')
    start_time = time.time()
    train_data_path = os.path.join(CURRENT_PATH, 'data/train')
    test_data_path = os.path.join(CURRENT_PATH, 'data/test')
    train_df = data_process.extract_feature(path_train, train_dtypes, save_path=train_data_path,  target='Y')
    # test_df = data_process.extract_feature(path_test, test_dtypes, save_path=test_data_path, data_process_params=params)
    print('time1:', time.time() - start_time)

    print('>>>[2]. Train Valid Data Split...')
    start_time = time.time()
    train_data, valid_data = train_test_split(train_df, test_size=0.2)
    print('time2:', time.time() - start_time)

    print('>>>[3]. Training Process...')
    start_time = time.time()
    predictors = ['TERMINALNO', 'TRIP_ID_max', 'LONGITUDE_var', 'LATITUDE_var', 'DIRECTION_var',\
                  'HEIGHT_var']
    target = 'Y'
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 1,
        'num_leaves': 3,  # 1400  # we should let it be smaller than 2^(max_depth)
        'max_depth': 2,  # -1 means no limit
        'min_child_samples': 5,  # Minimum number of data need in a child(min_data_in_leaf)
        'subsample': .8,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'verbose': 1
    }

    model = lgb_modelfit_nocv(params, train_data, valid_data, predictors, target)
    print('time3:', time.time() - start_time)


    print('>>>[4]. Test Data predict...')
    start_time = time.time()
    result = model.predict(valid_data[predictors])
    def f(x):
        return x if x>=0 else 0
    result = [f(i) for i in result]
    pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    pred_csv['Id'] = valid_data['TERMINALNO']
    pred_csv['Pred'] = result
    pred_csv.to_csv(path_test_out + 'pred.csv', index=False)
    print('time4:', time.time() - start_time)




if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()