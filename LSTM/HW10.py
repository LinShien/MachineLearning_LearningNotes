# -*- coding: utf-8 -*-
"""
Created on Tue May 29 23:29:21 2018

@author: user1
"""

'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
#from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80          # cut texts after this number of words (among top max_features most common words)
batch_size = 32 * 2      #一次訓練多少data

print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features) #考慮前20000個常用字元，將其他的非常用字編碼為oov_char
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


#print(x_train)                 #非narray object 其中包含list object
#print(y_train.shape)           


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen = maxlen)      #超過maxlen的就截掉, 少於的就補0
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)
print('x_train shape:', x_train.shape)                          #換成25000x80的matrix
print('x_test shape:', x_test.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))                         #生成20000x128的矩陣來encoding, 而在訓練的過程當中更新這個矩陣的值
model.add(LSTM(128, dropout=0.2, recurrent_dropout = 0.2, return_sequences=True))      #輸出為128units(看成一層的神經元數量)
model.add(LSTM(128, dropout=0.2, recurrent_dropout = 0.2))      #輸出為128units(看成一層的神經元數量)
model.add(Dense(1, activation='sigmoid'))


# try using different optimizers and different optimizer configs
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = 8, verbose = 1, 
          validation_data = (x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size = batch_size)
print('Test score:', score)
print('Test accuracy:', acc)