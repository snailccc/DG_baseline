import pandas as pd
from model.get_feature_method import *

FEATURE = ['article','word_seg']
if __name__ == '__main__':
    df = pd.read_csv('../local/local_train.csv').drop('word_seg',axis=1)
    ngram_feature(df,2,'article','train')
    print('train_article is finished')
