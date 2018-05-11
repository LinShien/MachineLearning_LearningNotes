# -*- coding: utf-8 -*-
"""
Created on Wed May  9 02:30:43 2018

@author: user1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:26:48 2018

@author: user1
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, PReLU, ZeroPadding2D
from keras import backend as K
import numpy as np

valid_num = 5000
batch_size = 200
num_classes = 10
epochs = 10

data_x = np.load('images_train.npy')
data_y = np.load('labels_train.npy')
#print(data_x.shape)       #50000x32x32x3   channels first
#print(data_y.shape)       #50000x10
#print(type(data_x))       #ndarray object 

x_test = np.load('images_test.npy')
x_train = data_x[valid_num:, :]

y_test = np.load('labels_test.npy')
y_train = data_y[valid_num :, :]

input_shape = x_train.shape[1], x_train.shape[2], x_train.shape[3]
print(input_shape)

model = Sequential()
model.add(ZeroPadding2D((1, 1),input_shape = input_shape))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())                         #BatchNorm    
model.add(PReLU()) 
model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(32, (3, 3)))   
model.add(BatchNormalization())                         #BatchNorm    
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))               #2
model.add(Dropout(0.5))   #--------
model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(64, (3, 3)))   
model.add(BatchNormalization())                         #BatchNorm    
model.add(PReLU())
model.add(ZeroPadding2D((1,1)))
#model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))   
model.add(BatchNormalization())                         #BatchNorm    
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))               #3
model.add(Dropout(0.5))   #--------
model.add(Flatten())

model.add(Dense(4096))               #4
model.add(BatchNormalization())                         #BatchNorm
model.add(PReLU())
model.add(Dropout(0.25))

#model.add(Dense(2048))                #4
#model.add(BatchNormalization())                        #BatchNorm
#model.add(PReLU())
#model.add(Dropout(0.5))

model.add(Dense(1000))                #4
model.add(BatchNormalization())                         #BatchNorm
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))     #5

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size = batch_size,
#          epochs = epochs,
#          verbose = 1,
#          validation_data = (x_test, y_test))

outfile = 'cnn_model_with_TS45_BS200_EP20withDropout'
model.load_weights(outfile)

score2 = model.evaluate(x_test, y_test, verbose = 1)

print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

