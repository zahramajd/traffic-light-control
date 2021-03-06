import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from pandas import DataFrame
from pandas import concat
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import glob
import os
from tensorflow.keras.utils import to_categorical


def load_data(path):

    # data = pd.DataFrame()
    # files = glob.glob(path+'/*.csv')

    # dflist = []
    # for file in files:
    #     df = pd.read_csv(file, header=None)
    #     dflist.append(df)

    # data = pd.concat([dflist[0],dflist[1]], axis = 0)

    data = pd.read_csv("./DataSet-1/dataset536.csv")
    info = data.iloc[:,0:8]
    states = data.iloc[:,8:9]
    phases = data.iloc[:,9:13]

    return info, states, phases

def convert_actions(phases):

    actions = pd.DataFrame(columns=['A'])
    
    for index, row in phases.iterrows():

        if(np.array_equal([33, 33, 13, 13],row.values)):
            actions.loc[index] = [0]

        if(np.array_equal([33, 13, 33, 13],row.values)):
            actions.loc[index] = [1]

        if(np.array_equal([33, 13, 13, 33],row.values)):
            actions.loc[index] = [2]

        if(np.array_equal([13, 33, 33, 13],row.values)):
            actions.loc[index] = [3]

        if(np.array_equal([13, 33, 13, 33],row.values)):
            actions.loc[index] = [4]

        if(np.array_equal([13, 13, 33, 33],row.values)):
            actions.loc[index] = [5]

        if(np.array_equal([33, 23, 23, 13],row.values)):
            actions.loc[index] = [6]

        if(np.array_equal([33, 23, 13, 23],row.values)):
            actions.loc[index] = [7]

        if(np.array_equal([33, 13, 23, 23],row.values)):
            actions.loc[index] = [8]

        if(np.array_equal([23, 33, 23, 13],row.values)):
            actions.loc[index] = [9]

        if(np.array_equal([23, 33, 13, 23],row.values)):
            actions.loc[index] = [10]

        if(np.array_equal([13, 33, 23, 23],row.values)):
            actions.loc[index] = [11]

        if(np.array_equal([23, 23, 33, 13],row.values)):
            actions.loc[index] = [12]

        if(np.array_equal([23, 13, 33, 23],row.values)):
            actions.loc[index] = [13]

        if(np.array_equal([13, 23, 33, 23],row.values)):
            actions.loc[index] = [14]

        if(np.array_equal([23, 23, 13, 33],row.values)):
            actions.loc[index] = [15]

        if(np.array_equal([23, 13, 23, 33],row.values)):
            actions.loc[index] = [16]

        if(np.array_equal([13, 23, 23, 33],row.values)):
            actions.loc[index] = [17]

        if(np.array_equal([23, 23, 23, 23],row.values)):
            actions.loc[index] = [18]

        if(np.array_equal([53, 13, 13, 13],row.values)):
            actions.loc[index] = [19]

        if(np.array_equal([13, 53, 13, 13],row.values)):
            actions.loc[index] = [20]

        if(np.array_equal([13, 13, 53, 13],row.values)):
            actions.loc[index] = [21]

        if(np.array_equal([13, 13, 13, 53],row.values)):
            actions.loc[index] = [22]

    return actions

def normalize(raw_info):
    min_max_scaler = preprocessing.MinMaxScaler()
    info = min_max_scaler.fit_transform(raw_info)
    return info

def split_data(data):
    
    train_percent = 0.9

    m = len(data)

    train = data[:int(train_percent * m)]
    test = data[int(train_percent * m):]

    return train, test

def state_state_1_timestep():
    train_X = info_train[:, :-1]
    test_X = info_test[:, :-1]

    train_y = states_train
    test_y = states_test

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def state_state_n_timestep(info, states, timesteps):

    n_features = 8
    n_obs = timesteps * n_features

    data = DataFrame(info)
    label = DataFrame(states)

    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    for i in range(0,1):
        cols.append(label.shift(-i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    # split to train & test
    train_percent = 0.9
    m = len(data)
    point = int(train_percent * m) 
    train = data[:point, :]
    test = data[point:, :]

    train_X = train[:, :n_obs]
    test_X = test[:, :n_obs]

    train_y = train[:, -1]
    test_y = test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # model = Sequential()
    # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # train_y = to_categorical(train_y)
    # test_y = to_categorical(test_y)
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(24, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# # fit network
    # model.fit(train_X, train_y, epochs=100, batch_size=72, verbose=0)
    # _, accuracy = model.evaluate(test_X, test_y, batch_size=72, verbose=0)

    # print(accuracy)

    return model

def state_action_state_n_timestep(info, actions, states, timesteps):

    n_features = 9
    n_obs = timesteps * n_features

    data = pd.concat([DataFrame(info), DataFrame(actions)], axis=1)
    label = DataFrame(states)

    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    for i in range(0,1):
        cols.append(label.shift(-i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    # split to train & test
    train_percent = 0.9
    m = len(data)
    point = int(train_percent * m) 
    train = data[:point, :]
    test = data[point:, :]

    train_X = train[:, :n_obs]
    test_X = test[:, :n_obs]

    train_y = train[:, -1]
    test_y = test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


    # model = Sequential()
    # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # train_y = to_categorical(train_y)
    # test_y = to_categorical(test_y)
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(24, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# # fit network
    # history =model.fit(train_X, train_y, epochs=70, batch_size=72, verbose=0)
    # _, accuracy = model.evaluate(test_X, test_y, batch_size=72, verbose=0)
    # print(accuracy)

    return model

def state_action_state_action_n_timestep(info, actions, states, timesteps):
    
    n_features = 9
    n_obs = timesteps * n_features

    data = pd.concat([DataFrame(info), DataFrame(actions)], axis=1)
    label = pd.concat([DataFrame(states), DataFrame(actions)], axis=1)


    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    for i in range(0,1):
        cols.append(label.shift(-i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    # split to train & test
    train_percent = 0.9
    m = len(data)
    point = int(train_percent * m) 
    train = data[:point, :]
    test = data[point:, :]

    train_X = train[:, :n_obs]
    test_X = test[:, :n_obs]

    train_y = train[:, -2:]
    test_y = test[:, -2:]

    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))


    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(2))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
    return model

def save_model(model, name):
    model_json = model.to_json()

    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(name + ".h5")
    return

def load_saved_model(name):

    # load json and create model
    json_file = open(name+'.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(name + ".h5")

    return load_model



raw_info, states, phases = load_data(path='./DataSet-1')
actions = convert_actions(phases)
info = normalize(raw_info)

model_num = 3
timesteps = 5

# save models
if(model_num == 1):
    model = state_state_n_timestep(info, states, timesteps=timesteps)
    save_model(model, name='state_state')

if(model_num == 2):
    model = state_action_state_n_timestep(info, actions, states, timesteps=timesteps)
    save_model(model, name='state_action_state')

if(model_num == 3):
    model = state_action_state_action_n_timestep(info, actions, states, timesteps=timesteps)
    save_model(model, name='state_action_state_action')



# run models on test data
raw_info, states, phases = load_data(path='./DataSet-2')
actions = convert_actions(phases)
info = normalize(raw_info)


if(model_num == 1):

    n_features = 8
    n_obs = timesteps * n_features

    data = DataFrame(info)
    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    test_X = data[:, :n_obs]
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))


    loaded_model = load_saved_model(name='state_state')
    yhat = model.predict(test_X)
    print(yhat)
    np.savetxt("state_state.csv", yhat, delimiter=",")

if(model_num == 2):

    n_features = 9
    n_obs = timesteps * n_features

    data = pd.concat([DataFrame(info), DataFrame(actions)], axis=1)
    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    test_X = data[:, :n_obs]
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))

    loaded_model = load_saved_model(name='state_action_state')
    yhat = model.predict(test_X)
    np.savetxt("state_action_state.csv", yhat, delimiter=",")

if(model_num == 3):
    n_features = 9
    n_obs = timesteps * n_features

    data = pd.concat([DataFrame(info), DataFrame(actions)], axis=1)
    cols = []
    for i in range(timesteps, 0, -1):
        cols.append(data.shift(i))

    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    data = data.values

    test_X = data[:, :n_obs]
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))

    loaded_model = load_saved_model(name='state_action_state_action')
    yhat = model.predict(test_X)
    np.savetxt("state_action_state_action.csv", yhat, delimiter=",")
