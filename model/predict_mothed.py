import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn import tree, svm, neighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
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

def svm_vote_predict(trainX, testX, test_id, label):
    test_svc = np.zeros((testX.shape[0], 5))
    kf = StratifiedKFold(n_splits=10, random_state=233, shuffle=True)
    for i, (train_index, val_index) in enumerate(kf.split(trainX, label)):
        if i > 4:
            break
        X_train = trainX[train_index]
        y_train = label[train_index]
        X_val = trainX[val_index]
        y_val = label[val_index]

        svc_clf = svm.LinearSVC(class_weight='balanced')
        svc_clf.fit(X_train, y_train)
        pred_val = svc_clf.predict(X_val)
        print(f1_score(y_val, pred_val, average='macro'))

        test_svc[:, i] = svc_clf.predict(testX)

    res_bag = []
    for r in test_svc:
        res_bag.append(int(Counter(r).most_common(1)[0][0]))

    res = pd.DataFrame({"class": res_bag, 'id': test_id})
    res['class'] = (res['class'] + 1).astype(int)
    res[["id", "class"]].to_csv('data/submission/sub_svm1.csv', index=None)

def LR_predict(trainX, testX, test_id, label, seed=0):
    test_lr = np.zeros((testX.shape[0], 19))
    kf = StratifiedKFold(n_splits=10, random_state=233, shuffle=True)
    for train_index, val_index in kf.split(trainX, label):
        X_train = trainX[train_index]
        y_train = label[train_index]
        X_val = trainX[val_index]
        y_val = label[val_index]

        seed += 10
        clf = LogisticRegression(C=4, dual=True, n_jobs=4, multi_class='ovr', random_state=seed)
        clf.fit(X_train, y_train)
        predi = clf.predict(X_val)
        print(f1_score(y_val, predi, average='macro'))
        res = clf.predict_proba(testX)
        test_lr += res
        # print(classification_report(y_val, predi, target_names=['%s' % i for i in range(20)]))

        preds = np.argmax(test_lr, axis=1)
        preds = preds + 1

        res = pd.DataFrame({"class": preds, 'id': test_id})
        res['class'] = res['class'].astype(int)
        res[["id", "class"]].to_csv('data/submission/sub_lr1.csv', index=None)
