import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    test_x = pd.read_csv('../data/encode_data/test_set_low_encoded.csv')
    train = pd.read_csv('../data/encode_data/train_set_low_encoded.csv')
    print(train)

    clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=66, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.3, min_child_weight=50, random_state=2019, n_jobs=-1
    )
    train_x=train
    train_y = train_x.pop('class')
    print(train_x.keys())
    os.system('pause')
    print(test_x.shape)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc')
    test_y = clf.predict_proba(test_x)
    label = []
    for it in test_y:
        temp = it.tolist()
        label_one = temp.index(max(temp))
        label.append(label_one+1)
    label = pd.DataFrame(label,columns=['class'])
    res = test_x[['id']]
    print(res.shape[0], label.shape[0])
    res = res.join(label)
    print(res.info)
    print('predict finish')
