import pandas as pd
from model.reduce_dimension_mothed import reduce_dimension

if __name__ == '__main__':
    df = pd.read_csv('../data/features/article.csv')
    articles = []
    for i in range(df.shape[0]):
        articles.append(df.iloc[i,0])
    articles = [str(i) for i in articles]
    df = pd.read_csv('../data/features/word_seg.csv')
    word_seg = []
    for i in range(df.shape[0]):
        word_seg.append(df.iloc[i,0])
    word_seg = [str(i) for i in word_seg]
    reduce_dimension('test10','train',articles,word_seg)