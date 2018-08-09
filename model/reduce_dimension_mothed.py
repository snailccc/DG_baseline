import pandas as pd

FEATURES = ['article','word_seg']

def reduce_dimension(name,set_type,articles,word_segs):
    if set_type == 'train':
        df = pd.read_csv('../data/splited_data/split_to_classes/{0}.csv'.format(name))
        classes = df.pop('class')
    elif set_type == 'test':
        df = pd.read_csv('../data/data_raw/test_set.csv')
    res = []
    print(articles)
    for i in range(df.shape[0]):
        item = []
        id = df.iloc[i,0]
        article = df.iloc[i,1].split(' ')
        word_seg = df.iloc[i,2].split(' ')
        temp=[]
        for it in article:
            if it in articles:
                print(it)
                temp.append(it)
        article = ' '.join(temp)
        temp = []
        for it in word_seg:
            if it in word_segs:
                temp.append(it)
        word_seg = ' '.join(temp)
        item.append(id)
        item.append(article)
        item.append(word_seg)
        res.append(item)
        if i %100 == 0 and i != 0:print(i)
    res = pd.DataFrame(res,columns=['id','article','word'])
    if set_type == 'train':
        res = res.join(classes)

    res.to_csv('../data/data_low_dimension/{0}_low.csv'.format(name),index=False)
    print('{0} is encoded'.format(name))



