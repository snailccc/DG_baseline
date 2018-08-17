import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

def lgb_predict(train_x,train_y,test_x,test_y=None):
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=66, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.3, min_child_weight=50, random_state=2019, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc')
    res = clf.predict_proba(test_x)
    label = []
    print(res)
    for it in res:
        temp = it.tolist()
        label_one = temp.index(max(temp))
        label.append(label_one + 1)
    label = pd.DataFrame(label, columns=['class'])
    res = test_x[['id']].reset_index().drop('index',axis=1)
    res = res.join(label)
    print('predict finish')
    return res

def svm_predict(train_x,train_y,test_x,test_y=None):
    model = SVC(c=1.0,kernel='rbf',gamma='auto')
    model.fit(train_x,train_y)
    res = model.predict_proba(test_x)
    label = []
    for it in res:
        temp = it.tolist()
        one = temp.index(max(temp))
        label.append(one + 1)
    label = pd.DataFrame(label,columns=['class'])
    res = test_x[['id']].reset_index().drop('index', axis=1)
    res = res.join(label)
    print('svm predict finish')
    return res

def RNN_predict(train_x,train_y,test_x,test_y=None):
    model = MLPClassifier(activation='relu',solver='adam',alpha=0.0001)
    model.fit(train_x,train_y)
    res = model.predict_proba(test_x)
    label = []
    for it in res:
        temp = it.tolist()
        one = temp.index(max(temp))
        label.append(one+1)
    label = pd.DataFrame(label,columns=['class'])
    res = test_x[['id']].reset_index().drop('index', axis=1)
    print('RNN predict finish')
    return res