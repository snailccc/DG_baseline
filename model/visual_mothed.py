import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

def class_article_visual(df):
    data=[0 for i in range(19)]
    xlabel = [str(x+1) for x in range(19)]
    n = df.shape[0]
    for i in range(n):
        classes = df.iloc[i,2].split(' ')
        classes = [int(x) for x in classes]
        for it in classes:
            data[it-1] += 1
    plt.ylim(0,200)
    plt.bar(range(len(data)),data,tick_label=xlabel)
    plt.savefig('../pic/word_seg(0.25_10).png')

# def test_feature_visual(df):
#     data =

