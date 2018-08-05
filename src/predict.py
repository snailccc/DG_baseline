import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    res = []
    probability = []

    data = pd.read_csv('../encode_data/train_encoded.csv')
    n = data.shape[0]
    class_test = data.iloc[n * 8 / 9:, :].pop('class')

    for i in range(1, 20):
        print('分类 %d 中。。。' % i)
        data = pd.read_csv('../data/splited_data/train_%s_splited.csv' % str(i))
        train_x = data.iloc[:n * 8 / 9, :]
        train_y = train_x.pop('class')
        test_x = data.iloc[n * 8 / 9:, :]
        test_y = test_x.pop('class')

        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=166, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.3, min_child_weight=50, random_state=2019, n_jobs=-1
        )
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc')
        res_temp = clf.predict_proba(test_x)[:1]

        if i == 1:
            probability = res_temp
            for j in range(n):
                j.append(1)
            res_temp = []
        else:
            for j in range(n):
                if probability[j] < res_temp[j]:
                    probability[j] = res_temp[j]
                    res[j] = i

    report=classification_report(class_test,res).split('\n')[3].strip()
    score=float(report.split(' ')[19])
    print(report)
    print(score)