import pandas as pd
import numpy as np
from model.visual_mothed import class_article_visual

if __name__ == '__main__':
    df = pd.read_csv('../data/features/word_seg.csv')
    class_article_visual(df)