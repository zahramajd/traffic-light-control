import tensorflow as tf
import pandas as pd
import numpy as np

#TODO: normalize
#TODO: split train & test
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
        f = False

        if(np.array_equal([33, 33, 13, 13],row.values)):
            actions.loc[index] = [0]
            f = True

        if(np.array_equal([33, 13, 33, 13],row.values)):
            actions.loc[index] = [1]
            f = True

        if(np.array_equal([33, 13, 13, 33],row.values)):
            actions.loc[index] = [2]
            f = True

        if(np.array_equal([13, 33, 33, 13],row.values)):
            actions.loc[index] = [3]
            f = True

        if(np.array_equal([13, 33, 13, 33],row.values)):
            actions.loc[index] = [4]
            f = True

        if(np.array_equal([13, 13, 33, 33],row.values)):
            actions.loc[index] = [5]
            f = True

        if(np.array_equal([33, 23, 23, 13],row.values)):
            actions.loc[index] = [6]
            f = True

        if(np.array_equal([33, 23, 13, 23],row.values)):
            actions.loc[index] = [7]
            f = True

        if(np.array_equal([33, 13, 23, 23],row.values)):
            actions.loc[index] = [8]
            f = True

        if(np.array_equal([23, 33, 23, 13],row.values)):
            actions.loc[index] = [9]
            f = True

        if(np.array_equal([23, 33, 13, 23],row.values)):
            actions.loc[index] = [10]
            f = True

        if(np.array_equal([13, 33, 23, 23],row.values)):
            actions.loc[index] = [11]
            f = True

        if(np.array_equal([23, 23, 33, 13],row.values)):
            actions.loc[index] = [12]
            f = True

        if(np.array_equal([23, 13, 33, 23],row.values)):
            actions.loc[index] = [13]
            f = True

        if(np.array_equal([13, 23, 33, 23],row.values)):
            actions.loc[index] = [14]
            f = True

        if(np.array_equal([23, 23, 13, 33],row.values)):
            actions.loc[index] = [15]
            f = True

        if(np.array_equal([23, 13, 23, 33],row.values)):
            actions.loc[index] = [16]
            f = True

        if(np.array_equal([13, 23, 23, 33],row.values)):
            actions.loc[index] = [17]
            f = True

        if(np.array_equal([23, 23, 23, 23],row.values)):
            actions.loc[index] = [18]
            f = True

        if(f == False):
            actions.loc[index] = [19]

    return actions

info, states, phases = load_data()
actions = convert_actions(phases)
