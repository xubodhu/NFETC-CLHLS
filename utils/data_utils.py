import pandas as pd
import numpy as np
import tensorflow as tf
import csv


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def load(data_file, filter=False):
    df = pd.read_csv(data_file, sep='\t', names=['p1', 'p2', 'words', 'mentions', 'types'], quoting=csv.QUOTE_NONE)
    words = np.array(df.words)
    mentions = np.array(df.mentions)
    positions = np.array([[x, y] for x, y in zip(df.p1, df.p2)])
    labels = np.array(df.types)
    if filter:
        one_label_ids =[]
        multi_label_ids =[]
        for i in range(labels.shape[0]):
            if " " not in labels[i]:
                one_label_ids.append(i)
            else:
                multi_label_ids.append(i)
        print('one label rate: ',len(one_label_ids)/labels.shape[0])
        print('multi label rate: ',len(multi_label_ids)/labels.shape[0])
        return words,mentions,positions,labels,one_label_ids,multi_label_ids
    else:
        return words, mentions, positions, labels


def batch_iter(data, batch_size, num_epochs,select_ids=None, shuffle=True):
    data = np.array(data)
    data_size = len(data)

    for epoch in range(num_epochs):
        if shuffle:# shuffle one epoch data
            if select_ids is not None:
                shuffle_indices =np.random.permutation(np.array(select_ids))
            else:
                shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        print('epoch data size:',len(shuffled_data))
        num_batches_per_epoch = int((len(shuffled_data) - 1) / batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(shuffled_data))
            yield shuffled_data[start_index:end_index]

def batch_iter2(data, batch_size, num_epochs,select_ids=None, p=None):
    data = np.array(data)
    data_size = len(data)

    for epoch in range(num_epochs):
        if select_ids is not None:
            new_p = p[select_ids]
        else:
            new_p=p
        new_p = new_p*-1
        batch_ids = np.random.choice(select_ids,batch_size,replace=False,p=softmax(new_p)).tolist()
        return data[batch_ids]

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()