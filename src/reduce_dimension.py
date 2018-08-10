import pandas as pd
from model.reduce_dimension_mothed import reduce_dimension

if __name__ == '__main__':
    df = pd.read_csv('../data/features/article.csv')
    articles = []
    for i in range(df.shape[0]):
        articles.append(df.iloc[i,0])

    df = pd.read_csv('../data/features/word_seg.csv')
    word_seg = []
    for i in range(df.shape[0]):
        word_seg.append(df.iloc[i,0])

    reduce_dimension('train_set_1','train',articles,word_seg)