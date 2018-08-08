import pandas as pd
from model.get_feature_method import max_num_feature
from model.get_feature_method import get_classes_feature,count_feature_classes

FEATURE = ['article','word_seg']
if __name__ == '__main__':
    for fea in FEATURE:
        for i in range(1,20):
            get_classes_feature('train_set_%d' % i,fea)
        count_feature_classes('train_set',fea)