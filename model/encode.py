import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

class Encode(object):
    @classmethod
    def cvEncoding(cls, df, features):
        cv = CountVectorizer()
        for feature in features:
            df[feature] = df[feature].astype('str')
            cv.fit(df[feature])
            res = cv.transform(df[feature]).toarray()
            names = []
            for it in cv.get_feature_names():
                names.append(feature + '_' + str(it))
            res = pd.DataFrame(res, columns=names)
            df = df.drop(feature, axis=1)
            df = df.join(res)
            print(feature + '  is finish')
        return df

    @classmethod
    def TfEncoding(cls,df,features,stop_word=None):
        Tfi = TfidfVectorizer()
        for feature in features:
            df[feature] = df[feature].astype('str')
            Tfi.fit(df[feature],stop_word=stop_word)
            res = Tfi.transform(df[feature]).toarray()
            names = []
            for it in Tfi.get_feature_names():
                names.append(feature+'_'+str(it))
            res = pd.DataFrame(res,columns=names)
            df = df.drop(feature,axis=1)
            df = df.join(res)
            print(feature + '  is finish')
        return df

    @classmethod
    def get_spares_matrix(train, test):
        label = train['class'].values
        test_id = test['id'].values
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8, use_idf=1, smooth_idf=1, sublinear_tf=1)
        trainX = vec.fit_transform(train['word_seg'].values.astype('U'))
        testX = vec.transform(test['word_seg'].values.astype('U'))

        cv = CountVectorizer(max_features=500000, max_df=0.7, min_df=3, lowercase=False, ngram_range=(2, 3))
        trainX_cv = cv.fit_transform(train['word_seg'].values.astype('U'))
        testX_cv = cv.transform(test['word_seg'].values.astype('U'))

        trainX = hstack([trainX_cv, trainX])
        testX = hstack([testX_cv, testX])

        scaler = StandardScaler(with_mean=False)
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        ch2 = SelectKBest(chi2, k=150)
        trainX = ch2.fit_transform(trainX, label)
        testX = ch2.transform(testX)

        return trainX, testX, test_id, label




