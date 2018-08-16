'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000        #常用字彙改成5000個
maxlen = 400               #字串長度改成400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 8

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,                 # max_features = input_dim(字彙表大小)而非data總量
                    embedding_dims,               # output vector dim of the embedding layer(編碼的維度)
                    input_length = maxlen))       # the dim of the input to this layer
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,                        # filter輸出的張數(跟2D一樣)  
                 kernel_size,                     # filter_length
                 padding='valid',                # 不做zero padding
                 activation='relu',
                 strides=1))                     #convolution走的步長
# we use max pooling:
model.add(GlobalMaxPooling1D())                  #對於時間上的data做maxpooling, 一個

# We add a vanilla hidden layer:
#dim of the input must equal to the dim of the length of the output of the GlobalMaxPooling1D layer
model.add(Dense(hidden_dims))                    
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs=epochs, verbose = 1, 
          validation_data=(x_test, y_test))
