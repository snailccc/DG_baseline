import pandas as pd
from model.reduce_dimension_mothed import reduce_dimension

if __name__ == '__main__':
    df = pd.read_csv('data/features/article.csv')
    articles = []
    for i in range(df.shape[0]):
        articles.append(df.iloc[i,0])
    articles = [str(i) for i in articles]

    df = pd.read_csv('data/features/word_seg.csv')
    word_segs = []
    for i in range(df.shape[0]):
        word_segs.append(df.iloc[i,0])
    word_segs = [str(i) for i in word_segs]

#    df = pd.read_csv('data/data_raw/train_set.csv')
#   reduce_dimension('train_set','train',df,articles,word_segs)
    df = pd.read_csv('data/data_raw/test_set.csv')
    reduce_dimension('test_set','test',df,articles,word_segs)
