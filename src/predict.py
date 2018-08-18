import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.predict_mothed import lgb_predict

if __name__ == '__main__':
    test_x = pd.read_csv('../data/encode_data/test_set_low_encoded.csv')
    train = pd.read_csv('../data/encode_data/train_set_low_encoded.csv')
    res = lgb_predict(train,test_x)
    res.to_csv('data/submission/sub_{0}.csv',index=False)
