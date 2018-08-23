import pandas as pd
import numpy as np
from model.encode import Encode

FEATURES='article,word'.split(',')

def get_encoded_data(name,data_type,df):
    if data_type == 'train':
        label = df.pop('class')
    #编码函数
    data=Encode.TfEncoding(df,FEATURES)
    if data_type == 'train' : data=data.join(label)
    data.to_csv('../data/encode_data/{0}_encoded.csv'.format(name),index=False)
    print('{0}_encoded saving complete'.format(name))

if __name__ == '__main__' :
    print('start encoding')
    df = pd.read_csv('../data/data_low_dimension/local_train_low.csv')
    get_encoded_data('local_train_low','train',df)