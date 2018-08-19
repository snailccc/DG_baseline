import pandas as pd

def split_data_to_TF(name):
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

def split_data_to_classes(name):
    df = pd.read_csv('../data/data_raw/{0}.csv'.format(name))
    n = 19
    for i in range(1,n+1):
        new_data = df[df['class'] == i]
        print(new_data)
        print('class %d are splited' % i)
        new_data.to_csv('../data/splited_data/split_to_classes/train_set_%d.csv'%i,index=False)