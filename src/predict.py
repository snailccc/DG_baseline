import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.predict_mothed import lgb_predict

if __name__ == '__main__':
    test_x = pd.read_csv('../data/encode_data/test_set_low_encoded.csv')
    train_x = pd.read_csv('../data/encode_data/train_set_low_encoded.csv')
    train_y = train_x.pop('class')
    res = lgb_predict(train_x,train_y,test_x)
    res.to_csv('data/submission/sub_4.csv',index=False)
