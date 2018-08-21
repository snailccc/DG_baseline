import pandas as pd
import operator
import os

def max_num_feature(name):
    df = pd.read_csv('data/data_raw/{0}.csv'.format(name))
    features = set()
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
                features.add(k)
        if i % 100 == 0:
            print('%d are finished , ' % i, 'the total is %d' % len(features))

    print('there are %d features in total' % len(features))
    features = list(features)
    features.sort(reverse=True)
    with open('data/article.txt', 'w') as f:
        for it in features:
            f.write(it)
            f.write('\n')
    print(len(features))

def max_frequecny_feature(name):
    df = pd.read_csv('../data/data_raw/{0}.csv'.format(name))
    features = {}
    for i in range(df.shape[0]):
        article = df.iloc[i, 1]
        L = list(set(article.split(' ')))
        for it in L:
            if it in features:
                features[it] += 1
            else:
                features[it] = 1
        if i % 100 == 0:
            print('%d data are dealed' % i)

    print('count finished')

    features = sorted(zip(features.values(), features.keys()), reverse=True)

    with open('data/features/article_frequency.txt') as f:
        for it in features:
            f.write(it[1], ' ', it[0])
            f.write('\n')

    print(len(features))

def get_classes_feature(name, feature):
    df = pd.read_csv('../data/splited_data/split_to_classes/{0}.csv'.format(name))
    features_num = {}
    features_fre = {}

    if feature == 'article':
        loc = 1
    elif feature == 'word_seg':
        loc = 2

    n = df.shape[0]
    for i in range(n):
        features = set()
        item = df.iloc[i, loc]
        item = item.split(' ')
        for it in item:
            if it in features:
                features_num[it] += 1
            else:
                features.add(it)
                if it in features_num:
                    features_num[it] += 1
                else:
                    features_num[it] = 1

                if it in features_fre:
                    features_fre[it] += 1
                else:
                    features_fre[it] = 1

        if i % 100 == 0:
            print('%d are count' % i)

    res = []
    for it in features_fre.keys():
        temp = []
        temp.append(it)
        temp.append(features_fre[it]/n)
        temp.append(features_num[it])
        res.append(temp)
    # saving。。。
    res = pd.DataFrame(res, columns=['id', 'frequency', 'num'])
    res = res[res['frequency'] > 0.25]
    print(res[res['frequency']>0.25].shape)
    res.to_csv('../data/features/{0}/{0}_{1}.csv'.format(feature, name), index=False)
    print('{0} is finished'.format(name))

def count_feature_classes(name, feature):
    feature_dic = {}
    feature_class = {}
    for index in range(1, 20):
        df = pd.read_csv('../data/features/{2}/{2}_{0}_{1}.csv'.format(name, index, feature))
        for i in range(df.shape[0]):
            id = df.iloc[i, 0]
            if id in feature_dic:
                feature_dic[id] += 1
                feature_class[id].add(str(index))
            else:
                feature_dic[id] = 1
                feature_class[id] = set()
                feature_class[id].add(str(index))
        print('%d csv is count' % index)
    feature_list = []
    for k, v in feature_dic.items():
        print(k,v)
        feature_list.append([k, v,' '.join(list(feature_class[k]))])
    feature_df = pd.DataFrame(feature_list, columns=['id', 'frequency','classes'])
    feature_df = feature_df[feature_df['frequency'] < 10]
    print(feature_df.info())
    feature_df.to_csv('../data/features/{0}.csv'.format(feature), index=False)

def find_all_feature(train=pd.DataFrame(),test=pd.DataFrame()):
    feature_train_article=set()
    feature_train_word_seg = set()
    for i in range(train.shape[0]):
        artcile = train.iloc[i,1].split(' ')
        word_seg = train.iloc[i,2].split(' ')
        feature_train_article = feature_train_article | set(artcile)
        feature_train_word_seg = feature_train_word_seg | set(word_seg)
        if i%100 == 0 and i != 0:
            print(i)
    feature_test_word_seg = set()
    feature_test_article = set()
    print('train_article is %d'%len(feature_train_article))
    print('train_word_seg is %d' % len(feature_train_word_seg))
    with open('../data/feature/train_feature_article.txt','w') as f:
        for it in feature_test_article:
            f.write(it)
            f.write('\t')
    with open('../data/feature/train_feature_word_seg.txt','w') as f:
        for it in feature_test_word_seg:
            f.write(it)
            f.write('\t')
    print("train's feature are got")
    for i in range(test.shape[0]):
        artcile = test.iloc[i,1]
        word_seg = test.iloc[i,2]
        feature_test_article = feature_test_article | set(artcile)
        feature_test_word_seg = feature_test_word_seg | set(word_seg)
        if i%100 == 0 and i != 0:
            print(i)
    with open('../data/feature/test_feature_article.txt','w') as f:
        for it in feature_test_article:
            f.write(it)
            f.write('\t')
    with open('../data/feature/test_feature_word_seg.txt','w') as f:
        for it in feature_test_word_seg:
            f.write(it)
            f.write('\t')
    print("test's feature are got")
    print('test_article is %d'%len(feature_test_article))
    print('test_word_seg is %d' % len(feature_test_word_seg))

    print('*******************************')
    print('the comment feature of article is %d ' % len(feature_train_article-feature_test_article))
    print('the comment feature of word_seg is %d ' % len(feature_train_word_seg - feature_test_word_seg))
