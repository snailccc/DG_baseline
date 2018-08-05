import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


class Encode(object):
    @classmethod
    def cvEncoding(cls, df, features):
        cv = CountVectorizer()
        print('start cv....')
        for feature in features:
            df[feature] = df[feature].astype('str')
            cv.fit(df[feature])
            res = cv.transform(df[feature]).toarray()
            names = []
            for i in range(res.shape[1]):
                names.append(feature + '_' + str(i))
            res = pd.DataFrame(res, columns=names)
            df = df.drop(feature, axis=1)
            df = df.join(res)
            print(feature + '  is finish')
        return df


