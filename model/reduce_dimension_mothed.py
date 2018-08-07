import pandas as pd
def reduce_dimension(name):
    df = pd.read_csv('../data/data_raw/{0}_set.csv'.format(name))
    # df = pd.read_csv('../test.csv')
    feature=[]
    with open('../data/features/article.txt') as f:
        for it in f:
            feature.append(it.split('\n')[0])
    articles=[]
    for i in range(df.shape[0]):
        article = []
        temp=df.iloc[i,1]
        temp=temp.split(' ')
        for it in temp:
            if it in feature:
                article.append(it)
        articles.append(' '.join(article))
    print(articles)
    df.drop('article',axis=1,inplace=True)
    df.insert(1,'article',pd.DataFrame(articles))
    print(df.head())
    df.to_csv('../data/data_low_dimension/{0}_low_dimension.csv'.format(name),index=False)
    # df.to_csv('../res.csv',index=False)

