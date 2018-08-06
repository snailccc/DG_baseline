import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer

# from model.encode import Encoding


if __name__ == '__main__':
    df = pd.read_csv('data/train_set.csv')
    feature = set()
    article = df.iloc[1, 1]
    for i in range(df.shape[0]):
        article = df.iloc[i, 1]
        L = article.split(' ')
        count = {}
        for it in L:
            if it in count:
                count[it] += 1
            else:
                count[it] = 1
        for k, v in count.items():
            if v > 100:
                feature.add(k)
        if i % 100 == 0:
            print('%d are finished , ' % i,'the total is %d'%len(feature))

    print('there are %d features in total' % len(feature))
    feature=list(feature)
    feature.sort(reverse=True)
    with open('data/article.txt', 'w') as f:
        for it in feature:
            f.write(it)
    print(len(feature))