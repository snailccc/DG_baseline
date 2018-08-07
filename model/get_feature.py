import pandas as pd

def max_num_feature(name):
    df = pd.read_csv('data/data_raw/{0}.csv'.format(name))
    feature = set()
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
            print('%d are finished , ' % i, 'the total is %d' % len(feature))

    print('there are %d features in total' % len(feature))
    feature = list(feature)
    feature.sort(reverse=True)
    with open('data/article.txt', 'w') as f:
        for it in feature:
            f.write(it)
            f.write('\n')
    print(len(feature))

def max_frequecny_feature(name):
    df=pd.read_csv('../data/data_raw/{0}.csv'.format(name))
    features={}
    for i in range(df.shape[0]):
        article=df.iloc[i,1]
        L=list(set(article.split(' ')))
        for it in L:
            if it in features:
                features[it]  += 1
            else:
                features[it] = 1
        if i % 100 == 0:
            print('%d data are dealed'%i)

    print('count finished')

    features = sorted(zip(features.values(),features.keys()),reverse=True)

    with open('data/features/article_frequency.txt') as f:
        for it in features:
            f.write(it[1],' ',it[0])
            f.write('\n')

    print(len(features))

