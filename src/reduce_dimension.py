import pandas as pd

def reduce_dimension(name):
    df = pd.read_csv('../data/data_raw/{0}_set.csv'.format(name))
    feature=[]
    with open('../data/features/article.txt') as f:
        for it in f:
            feature.append(it)
    articles=[]
    for i in range(df.shape[0]):
        article = []
        temp=df.iloc[i,1]
        temp=temp.split(' ')
        for it in temp:
            if it in feature:
                article.append(it)
        articles.append(' '.join(article))
    df.drop('article',axis=1)
    df.insert(pd.DataFrame(articles),1)
    df.to_csv('../data/data_low_dimension/{0}_low_dimension.csv'.format(name),index=False)

if __name__ == '__main__':
    df=reduce_dimension('test')