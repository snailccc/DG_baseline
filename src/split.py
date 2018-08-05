import pandas as pd

def split_data(name):
    data=pd.read_csv('../data/{0}_set.csv'.format(name))
    label=data.pop('class')
    print(label.shape)
    for i in range(1,20):
        label_temp=[]
        for j in range(label.shape[0]):
            if label.iloc[j] == i:
                label_temp.append(1)
            else:
                label_temp.append(0)
        label_temp=pd.DataFrame(label_temp,columns=['label'])
        data_temp=data.join(label_temp)
        print(data_temp.info())
        data_temp.to_csv('../data/splited_data/name_{0}_splited.csv'.format(str(i)),index=False)
        data_temp=None
        label_temp=None
        print('name_{0} is finished'.format(str(i)))
    print('splited is ok')


if __name__ == '__main__':
    split_data('train')