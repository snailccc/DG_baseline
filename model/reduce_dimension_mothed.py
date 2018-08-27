import pandas as pd
import numpy as np
import os
import gc
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
FEATURES = ['article','word_seg']

def reduce_dimension(name,set_type,articles,word_segs):
    if set_type == 'train':
        #df = pd.read_csv('../data/splited_data/split_to_classes/{0}.csv'.format(name))
        df = pd.read_csv('../data/data_raw/{0}.csv'.format(name))
        classes = df.pop('class')
    elif set_type == 'test':
        df = pd.read_csv('../data/data_raw/test_set.csv')
    res = []
    for i in range(df.shape[0]):
        item = []
        id = df.iloc[i,0]
        article = df.iloc[i,1].split(' ')
        word_seg = df.iloc[i,2].split(' ')
        target = set(article)
        for it in target:
            if it not in articles:
                article = [x for x in article if x != it]
        article = ' '.join(article)
        target = set(word_seg)
        for it in target:
            if it not in word_segs:
                word_seg = [x for x in word_seg if x != it]
        word_seg = ' '.join(word_seg)
        item.append(id)
        item.append(article)
        item.append(word_seg)
        res.append(item)
        if i %100 == 0 and i != 0:print(i)
    res = pd.DataFrame(res,columns=['id','article','word_seg'])
    if set_type == 'train':
        res = res.join(classes)
    print(res.info())
    res.to_csv('../data/data_low_dimension/{0}_low.csv'.format(name),index=False)
    print('{0} is encoded'.format(name))

def pca_reduce(df,set_type='test',n_components=10):
    df_id = df.pop('id')
    if set_type == 'train':
        classes = df.pop('class')
    pca = PCA(n_components=n_components)
    res = pd.DataFrame(pca.fit_transform(df))
    if set_type == 'train':
        res = pd.concat([df_id,res,classes],axis=1)
    else:
        res = pd.concat([df_id, res], axis=1)
    return res

def get_set(reader=None, set_type=''):
    if set_type == 'train':
        set_type = ['id', 'article', 'word_seg', 'class']
    elif set_type == 'test':
        set_type = ['id', 'article', 'word_seg']
    else:
        return 'wrong'
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(200)[set_type]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    return pd.concat(chunks, axis=0)

def KB_Select(df, features, label):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.8, use_idf=1, smooth_idf=1, sublinear_tf=1)
    X = vec.fit_transform(df[features])
    feature = vec.get_feature_names()
    ch2 = SelectKBest(chi2, 150)
    X = ch2.fit_transform(X, label)
    feature = [feature[i] for i in ch2.get_support(indices=True)]
    return X, feature

def tf_idf_Select(train, test, feature):
    path = 'data/features/vocab_{0}.pk1'.format(feature)
    if os.path.exists(path):
        vocab = pickle.load(open(path, 'rb'))
    else:
        corpus = []
        for doc in train[feature].tolist() + test[feature].tolist():
            doc = doc.split()
            corpus.append(doc)
        word_to_doc = {}
        idf = {}
        total_doc_num = float(len(corpus))
        # doc_keys = list(word_to_doc.keys())
        for doc in corpus:
            for word in set(doc):
                if word not in word_to_doc:
                    word_to_doc[word] = 1
                else:
                    word_to_doc[word] += 1
        doc_keys = list(word_to_doc.keys())
        for word in doc_keys:
            if word_to_doc[word] > 10:
                idf[word] = np.log(total_doc_num / (word_to_doc[word] + 1))
        sort_idf = sorted(idf.items(), key=lambda x: x[1])
        vocab = [x[0] for x in sort_idf]
        pickle.dump(vocab, open(path, 'wb'))
    return vocab

def deal_raw_data(path):
    train = get_set(pd.read_csv('data/data_raw/local_train.csv', iterator=True), 'train')
    test = get_set(pd.read_csv('data/data_raw/local_test.csv', iterator=True), 'test')
    gc.collect()

    train = train.sample(frac=1)
    train['class'] = train['class'] - 1
    label = train['class'].values
    for feature in FEATURES:
        train_x, feature_names = KB_Select(train, feature, label)
        vocab = tf_idf_Select(train, test, feature)
        vocab = vocab[:13000]
        vocab = set(vocab) & set(feature_names)
        vocab_dict = {w: 1 for w in vocab}

        def filter_word(x):
            x = x.split()
            s = []
            for w in x:
                if w in list(vocab_dict.keys()):
                    s.append(w)
            return ' '.join(s)

        train[feature] = train[feature].map(lambda x: filter_word(x))
        test[feature] = test[feature].map(lambda x: filter_word(x))
    train.to_csv('data/data_low/train_low.csv', index=False)
    test.to_csv('data/data_low/test_low.csv', index=False)
    return train, test

