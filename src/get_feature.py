import pandas as pd
from model.get_feature_method import max_num_feature
from model.get_feature_method import get_classes_feature,count_feature_classes
from model.get_feature_method import *

FEATURE = ['article','word_seg']
if __name__ == '__main__':
    train = pd.read_csv('data/data_raw/train_set.csv')
    test = pd.read_csv('data/data_raw/test_set.csv')
    find_all_feature(train,test)
