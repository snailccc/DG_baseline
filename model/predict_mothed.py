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
    train_x = train_x.drop('id',axis=1)
    test_id = test_x.pop('id').reset_index().drop('index',axis=1)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=66, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.3, min_child_weight=50, random_state=2019, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc')
    label = clf.predict(test_x)
    label = pd.DataFrame(label, columns=['class'])
    res = test_id.join(label)
    print('lgb predict finish')
    return res

def svm_predict(train_x,train_y,test_x,test_y=None,max_iter=666):
    train_x = train_x.drop('id',axis=1)
    test_id = test_x.pop('id').reset_index().drop('index',axis=1)
    model = SVC(C=1.0,kernel='rbf',gamma='auto',max_iter=max_iter)
    train_x = train_x.fillna(-1)
    test_x = test_x.fillna(-1)
    print('拟合中')
    model.fit(train_x,train_y)
    print('预测中')
    label = model.predict(test_x)
    label = pd.DataFrame(label,columns=['class'])
    res = test_id.join(label)
    print('svm predict finish')
    return res

def RNN_predict(train_x,train_y,test_x,test_y=None):
    train_x = train_x.drop('id',axis=1)
    test_id = test_x.pop('id').reset_index().drop('index',axis=1)
    model = MLPClassifier(activation='relu',solver='adam',alpha=0.0001)
    model.fit(train_x,train_y)
    label = model.predict(test_x)
    label = pd.DataFrame(label,columns=['class'])
    res = test_id.join(label)
    print('RNN predict finish')
    return res

def bayes_predict(train_x,train_y,test_x,test_y=None):
    train_x = train_x.drop('id',axis=1)
    test_id = test_x.pop('id').reset_index().drop('index',axis=1)
    model = naive_bayes.GaussianNB()
    model.fit(train_x,train_y)
    label = model.predict(test_x)
    label = pd.DataFrame(label,columns=['class'])
    res = test_id.join(label)
    print('bayers predict finish')
    return res