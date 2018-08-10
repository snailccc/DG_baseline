import pandas as pd
import numpy as np
from model.encode import Encode

FEATURES='article,word'.split(',')

def get_encoded_data(name,data_type):
    df=pd.read_csv('../data/data_low_dimension/{0}.csv'.format(name))
    if data_type == 'train':
        label = df.pop('class')
    data=Encode.cvEncoding(df,FEATURES)
    if data_type == 'train' : data=data.join(label)
    data.to_csv('../data/encode_data/{0}_encoded.csv'.format(name),index=False)
    print('{0}_encoded saving complete'.format(name))

if __name__ == '__main__' :
    print('start cv')
    for i in range(1,20):
        get_encoded_data('train_set_%d_low'%i)
    get_encoded_data('test_low')