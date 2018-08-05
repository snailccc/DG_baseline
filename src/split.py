import pandas as pd

def split_data(name):
    data=pd.read_csv('../data/{0}_set.csv'.format(name))
    label=data.pop('class')
    for i in range(1,19):
        label_temp=[]
        for j in range(label.shape[0]):
            if label.iloc[j,0] == j:
                label_temp.append(1)
            else:
                label_temp.append(0)
        data_temp=data.join(label_temp)
        data_temp.to_csv('../data/splited_data/name_{0}_splited.csv'.format(i),index=False)
        print('name_{0} is finished'.format(i))
    print('splited is ok')


if __name__ == '__main__':
    split_data('train')