import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.disable_v2_behavior()
import scipy.io

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.layers import LSTM, Activation, CuDNNLSTM
# from keras.layers import LSTM, Activation, CuDNNLSTM
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

X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

X_pred = X_pred_map
y_pred_ref = y_pred_ref_map

data_dim = X_train.shape[2]  # number of input features
timesteps = X_train.shape[1]
num_classes = y_train.shape[2]  # number of output features
batch_size = 10

rms = RMSprop(lr=0.001, decay=0.0001)
adam = Adam(lr=0.001, decay=0.0001)
model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, data_dim)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
model.add(Activation('relu'))
# model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))  # 增加一层LSTM
# model.add(Activation('relu'))
# model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))  # 增加一层LSTM
# model.add(Activation('relu'))
# model.add(Dropout(0.2))   #
model.add(Dense(100))
# model.add(Activation('relu'))
model.add(Dense(num_classes))
model.summary()

model.compile(loss='mean_squared_error', 
              optimizer=adam,  
              metrics=['mse'])
best_loss = 0.1
train_loss = []
test_loss = []
history = []

with tf.device('/device:GPU:1'):

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    

    start = time.time()

    epochs = 3000
    for e in range(epochs):
        print('epoch = ', e + 1)

        Ind = list(range(len(X_data_new)))
        shuffle(Ind)
        ratio_split = 0.7
        Ind_train = Ind[0:round(ratio_split * len(X_data_new))]
        Ind_test = Ind[round(ratio_split * len(X_data_new)):]

        X_train = X_data_new[Ind_train]
        y_train = y_data_new[Ind_train]
        X_test = X_data_new[Ind_test]
        y_test = y_data_new[Ind_test]

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  # validation_split=0.2,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  epochs=1)
        score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        train_loss.append(score0[0])
        test_loss.append(score[0])

        if test_loss[e] < best_loss:
            best_loss = test_loss[e]
            model.save(dataDir + 'results/VTB_system(LSTM-f)/my_best_model.h5')

    end = time.time()
    running_time = (end - start)/3600
    print('Running Time: ', running_time, ' hour')


plt.figure()
plt.plot(np.array(train_loss), 'b-')
plt.plot(np.array(test_loss), 'm-')
plt.show()
# Save scaler
joblib.dump(scaler_X, dataDir+'results/scaler_X.save')
joblib.dump(scaler_y, dataDir+'results/scaler_y.save')

scipy.io.savemat(dataDir+'results/Model_MSE.mat',
                 {'train_loss': train_loss, 'test_loss': test_loss, 'best_loss': best_loss,
                  'running_time': running_time, 'epochs': epochs})
