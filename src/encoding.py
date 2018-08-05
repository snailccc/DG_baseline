import pandas as pd
import numpy as np
from model.encode import Encode

FEATURES='article,word_seg'.split(',')

def get_encoded_data(name):
    data=pd.read_csv('../data/{0}.csv'.format(name))
    if name == 'train_set':
        label = data.pop('class')
    data=Encode.cvEncoding(data,FEATURES)
    data=data.join(label)
    data.to_csv('../data/encode_data/{0}_encoded.csv'.format(name),index=False)
    print('{0}_encoded saving complete'.format(name))

if __name__ == '__main__' :
    get_encoded_data('test_set')
    get_encoded_data('train_set')