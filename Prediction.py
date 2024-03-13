# 基于深度学习代理模型LSTM的列车-有砟轨道动力性能预测模型

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from random import shuffle, randint
import time

# from sklearn.externals import joblib  # save scaler
# 解决版本兼容问题
tf.compat.v1.disable_eager_execution()
import joblib  # 使用joblib保存svr训练的模型 from joblib import dump, load

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;



# Load data
dataDir = 'H:/青藏线随机分析-TBTI和LSTM/列车-轨道系统DeepLSTM-QZX/QZX_LSTM/' # Replace the directory
mat = scipy.io.loadmat(dataDir+'data/QZX_LSTM_Train100.mat')

X_data = mat['input_tf']
y_data = mat['target_tf']
train_indices = mat['trainInd'] - 1
test_indices = mat['valInd'] - 1

# Scale data
X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], X_data.shape[2]])
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_X.fit(X_data_flatten)
X_data_flatten_map = scaler_X.transform(X_data_flatten)
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], X_data.shape[2]])

y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], y_data.shape[2]])
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y.fit(y_data_flatten)
y_data_flatten_map = scaler_y.transform(y_data_flatten)
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

# Unknown data
X_pred = mat['input_pred_tf']
y_pred_ref = mat['target_pred_tf']

# Scale data
X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])

y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

X_data_new = X_data_map
y_data_new = y_data_map


batch_size = 10
model_best = load_model(dataDir + 'results/my_best_model.h5')  #

data_dim = X_data.shape[2]  # number of input features
timesteps = X_data.shape[1]
num_classes = y_data.shape[2]  # number of output features

# 训练集
y_train_pred_all=np.zeros((len(train_indices[0]),timesteps,num_classes))
for i_Tnum in range(len(train_indices[0])):
    X_train = X_data_new[i_Tnum]
    X_train=np.reshape(X_train, [1, timesteps, data_dim])
    y_train = y_data_new[i_Tnum]
    y_train = np.reshape(y_train, [1, timesteps, num_classes])


    y_train_pred = model_best.predict(X_train)

    y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0] * y_train_pred.shape[1], y_train_pred.shape[2]])
    y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
    y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])
    # 每一个训练集加入整体
    y_train_pred_all[i_Tnum]=y_train_pred


# 测试集
Train_num=len(train_indices[0])
y_test_pred_all=np.zeros((X_data.shape[0]-len(train_indices[0]),timesteps,num_classes))
for i_Testnum in range(X_data.shape[0]-len(train_indices[0])):
    X_test = X_data_new[Train_num+i_Testnum]
    X_test = np.reshape(X_test, [1, timesteps, data_dim])
    y_test = y_data_new[Train_num+i_Testnum]
    y_test = np.reshape(y_test, [1, timesteps, num_classes])

    y_test_pred = model_best.predict(X_test)

    y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0] * y_test_pred.shape[1], y_test_pred.shape[2]])
    y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
    y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

    # 每一个训练集加入整体
    y_test_pred_all[i_Testnum] = y_test_pred



# 预测集
Pre_num=X_pred_map.shape[0]    # 预测集的数目
y_pure_pred_all = np.zeros((Pre_num,timesteps,num_classes))
for i_Prenum in range(Pre_num):
    X_pred = X_pred_map[i_Prenum]
    X_pred = np.reshape(X_pred, [1, timesteps, data_dim])
    y_pure_preds = model_best.predict(X_pred)

    y_pure_preds_flatten = np.reshape(y_pure_preds,[y_pure_preds.shape[0] * y_pure_preds.shape[1], y_pure_preds.shape[2]])
    y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
    y_pure_preds = np.reshape(y_pure_preds, [1, timesteps, num_classes])

    # 每一个预测集加入整体
    y_pure_pred_all[i_Prenum] = y_pure_preds


# Reverse map to original magnitude
X_train_orig = X_data[0:len(train_indices[0])]
y_train_orig = y_data[0:len(train_indices[0])]
X_test_orig = X_data[len(train_indices[0]):]
y_test_orig = y_data[len(train_indices[0]):]
X_pred_orig = mat['input_pred_tf']
y_pred_ref_orig = mat['target_pred_tf']

# Save scaler
joblib.dump(scaler_X, dataDir+'results/scaler_X.save')
joblib.dump(scaler_y, dataDir+'results/scaler_y.save')


scipy.io.savemat(dataDir+'results/VTB_QZX_Response.mat',
                 {'y_train_orig': y_train_orig, 'y_train_pred': y_train_pred_all,
                   'y_test_orig': y_test_orig, 'y_test_pred': y_test_pred_all,
                  'y_pred_ref': y_pred_ref, 'y_pred_ref_orig': y_pred_ref_orig, 'y_pure_preds': y_pure_pred_all,
                  'train_indices': train_indices[0], 'test_indices': test_indices[0]})
