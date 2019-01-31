import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing

#TODO: LSTM
#TODO: test


def load_data():
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
    
    train_percent = 0.8

    m = len(data)

    train = data[:int(train_percent * m)]
    test = data[int(train_percent * m):]

    return train, test


raw_info, states, phases = load_data()
actions = convert_actions(phases)
info = normalize(raw_info)


info_train, info_test = split_data(info)
states_train, states_test = split_data(states)
actions_train, actions_test = split_data(actions)
