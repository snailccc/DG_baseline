#encoding=utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.predict_mothed import lgb_predict,svm_predict,RNN_predict


if __name__ == '__main__':
    train = pd.read_csv('data/encode_data/train_set_low_encoded.csv',dtype={'class':np.int32})
    test = train.pop('class')
    train_x,test_x,train_y,test_y = train_test_split(train,test,test_size=0.2,random_state=166)
    #modal choose
    res = svm_predict(train_x,train_y,test_x)

    res = res[['class']]
    test_y = test_y.reset_index().drop('index',axis=1)
    report=classification_report(res,test_y)
    score = report.strip().split('\n')[-1]
    print(score)
