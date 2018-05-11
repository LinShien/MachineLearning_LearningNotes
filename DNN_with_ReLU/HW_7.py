# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:02:23 2018

@author: user1
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:20:58 2018

@author: user1
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, PReLU, ELU
from keras.utils import np_utils

batch_size = 100   #一次訓練的data個數
nb_classes = 10     #判斷的種類
nb_epoch = 5      #訓練週期(走完所有data算一個epoch)

#########################################data transformation#############################################
#print(mnist.load_data())
#print(type(mnist.load_data()))
#print(len(mnist.load_data()))          #data目前為len = 2的tuple data[0]為trainning sets, data[1]為testing sets
(X_trainning, Y_trainning), (X_testing, Y_testing) = mnist.load_data()

#print(X_trainning.shape)      #X_trainning 60000x28x28
#print(Y_trainning.shape)
#print(X_testing.shape)        #X_testing 10000x28x28

X_trainning = np.reshape(X_trainning, (60000, 28**2)).astype('float32')       #換成60000x764 且為float的矩陣, 並轉換精確度
X_testing = np.reshape(X_testing, (10000, 28**2)).astype('float32')          #換成10000x764 且為float的矩陣, 並轉換精確度

#keras.utils.to_categorical(y, num_classes=None) => Converts a class vector (integers) to binary class matrix.
Y_trainning = np_utils.to_categorical(Y_trainning, nb_classes)        #把Y換成60000x10x1的vectors
Y_testing = np_utils.to_categorical(Y_testing, nb_classes)
#print(Y_trainning.shape)

#########################################data transformation#############################################


model = Sequential()       #construct a NN(sequential object)
#print(type(model))
#keras.models.Dense(..) Just your regular densely-connected NN layer
model.add(Dense(input_dim = 28**2, units = 500, activation = 'sigmoid'))    #hidden units = 500
model.add(Dense(units=500, activation = 'sigmoid'))                #第2層hidden layer

model.add(Dense(units=500, activation = 'sigmoid'))
model.add(Dense(units=10, activation = 'sigmoid'))                 #output layer

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_trainning, Y_trainning,
                    batch_size = batch_size, epochs = nb_epoch,
                    verbose = 1, validation_data = (X_testing, Y_testing))   #verbose = Verbosity mode

score1 = model.evaluate(X_trainning, Y_trainning, verbose = 1)   #用此模型來測試理想準確度
score2 = model.evaluate(X_testing, Y_testing, verbose = 1)   #用此模型來測試理想準確度
print('epochs: %d, batch size: %d' %(nb_epoch, batch_size))
print('Test ideal cost:', score1[0])
print('Test ideal accuracy:%f ' %(score1[1] * 100))
print('Test actual cost:', score2[0])
print('Test actual accuracy:%f ' %(score2[1] * 100)) 