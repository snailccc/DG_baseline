import pandas as pd
import os
FEATURES = ['article','word_seg']

def reduce_dimension(name,set_type,articles,word_segs):
    if set_type == 'train':
        #df = pd.read_csv('../data/splited_data/split_to_classes/{0}.csv'.format(name))
        df = pd.read_csv('../data/data_raw/{0}.csv'.format(name))
        classes = df.pop('class')
    elif set_type == 'test':
        df = pd.read_csv('../data/data_raw/test_set.csv')
    res = []
    for i in range(df.shape[0]):
        item = []
        id = df.iloc[i,0]
        article = df.iloc[i,1].split(' ')
        word_seg = df.iloc[i,2].split(' ')
        target = set(article)
        for it in target:
            if it not in articles:
                article = [x for x in article if x != it]
        article = ' '.join(article)
        target = set(word_seg)
        for it in target:
            if it not in word_segs:
                word_seg = [x for x in word_seg if x != it]
        word_seg = ' '.join(word_seg)
        item.append(id)
        item.append(article)
        item.append(word_seg)
        res.append(item)
        if i %100 == 0 and i != 0:print(i)
        if i == 300 :
            break
    res = pd.DataFrame(res,columns=['id','article','word'])
    if set_type == 'train':
        res = res.join(classes)
    print(res.info())
    res.to_csv('../data/data_low_dimension/{0}_low.csv'.format(name),index=False)
    print('{0} is encoded'.format(name))



