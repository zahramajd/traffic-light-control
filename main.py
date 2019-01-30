import tensorflow as tf
import pandas as pd
import glob
import os

#TODO: normalize
#TODO: convert actions
#TODO: split train & test
#TODO: LSTM
#TODO: test

data = pd.read_csv("./DataSet-1/dataset536.csv")
info = data.iloc[:,0:8]
states = data.iloc[:,8:9]
actions = data.iloc[:,9:13]

