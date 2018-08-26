import pandas as pd
from model.get_feature_method import *

FEATURE = ['article','word_seg']

def tf_idf():
    for name in FEATURE:
        for i in range(1,20):
            df = pd.read_csv('../data/splited_data/split_to_classes/train_set_%d.csv'%i)
            res = get_tf(df,name)
            res.to_csv('../data/features/tf_idf/{0}/{0}_{1}.csv'.format(name,i),index=False)
        res = get_idf(name)
        res.to_csv('../data/features/tf_idf/{0}.csv'.format(name), index=False)


if __name__ == '__main__':
    tf_idf()
