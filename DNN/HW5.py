# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:13:23 2018

@author: user1
"""
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))  #np.exp() Calculate the exponential of all elements in the input array

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2) :
    m1 = X.shape[0]                # #of rows
    a1 = np.insert(X, 0, values = np.ones(m1), axis = 1)     #在每組data補上bias項
    z2 = a1 * theta1.T            #z2為5000x1之vector
    m2 = z2.shape[0]
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m2), axis= 1)
    z3 = a2 * theta2.T
    output = sigmoid(z3)
    return a1, z2, a2, z3, output

def computCost(params, input_size, hidden_size, num_labels, X, Y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    Y = np.matrix(Y)
    
    #reshape the theta matrix, cuz the dimensions of hidden_size and input_size is unknown 
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    a1, z2, a2, z3, output = forward_propagate(X, theta1, theta2)
    
    J = 0
    for i in range(m):
        first_term = np.multiply(-Y[i, :], np.log(output[i, :]))      #計算y(i)*logh(i)
        second_term = np.multiply(Y[i, :] - 1, np.log(1 - output[i, :]))
        J += np.sum(first_term + second_term)
        
    J /= m
    #with regulation term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J                                          #column從1開始是為了排除bias case

def back_propagate(params, input_size, hidden_size, num_labels, X, Y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    Y = np.matrix(Y)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    a1, z2, a2, z3, output = forward_propagate(X, theta1, theta2)
    
    #intializations
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    J = computCost(params, input_size, hidden_size, num_labels, X, Y, learning_rate)
    
    #back propagation
    for t in range(m):      #共有m組資料, 為求出m筆資料下平均的gradient, 最後可以用於gradient descent
        a1t = a1[t, :]      #有包含bias case   1x401
        z2t = z2[t, :]      #z2 1x25
        a2t = a2[t, :]      #a2 1x26
        outputt = output[t, :]    #1x10
        Yt = Y[t, :]        #y取一組資料1x10
        
        d3t = outputt - Yt        #output error, delta3   1x10 
        
        z2t = np.insert(z2t, 0, values = np.zeros(1))    #加入bias case才有辦法進行乘法
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))   #26x10 10x1相乘, 再跟1x26 elementwise相乘
        #Accumulate the gradient from this data set
        delta1 = delta1 + (d2t[:, 1:]).T * a1t           #bias case要排除 d2t為1x26維度,處理後變成25x1 , delta1為25x401維
        delta2 = delta2 + (d3t.T * a2t)                  #delta2 10x26 
        
    delta1 = delta1 / m      #求出平均微分值(gradient)
    delta2 = delta2 / m      #求出平均微分值(gradient)
    
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) /m   #bias case需要排除, 只需更新非bias項
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) /m
    
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))     #25*401+10*26
    
    return J, grad
        
data = loadmat('data/ex4data1.mat')
print(data['X'].shape, data['y'].shape)     #共5000比資料, 一筆xi資料大小為400x1的vector

y = data['y']
encoder = OneHotEncoder(sparse = False)
y_oneshot = encoder.fit_transform(data['y'])        #把y擴充成10x1的vector, 並把原本的值map成0和1
print(y_oneshot.shape)
#print(y_oneshot[0, :])                      #用slice檢查結果
#print(y.shape)

input_size = 400      #其實就是batch size
hidden_size = 50
num_labels = 10
learning_rate = 1
#np.random.random Return random floats narray in the half-open interval[0.0, 1.0)
params = (np.random.random(size = hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) 
#print(hidden_size * (input_size + 1) + num_labels * (hidden_size + 1))     params為10285x1vector
#print(params)

X = data['X']

m = X.shape[0]
X = np.matrix(X)

#theta1, theta2 用params來隨機取值,theta1大小要為25x(400+1) theta2為 10x(25+1)
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

#print(theta1.shape, theta2.shape)
#print(theta1[:, 1:])
#a1, z2, a2, z3, output = forward_propagate(X, theta1, theta2)
#print(a1.shape, z2.shape, a2.shape, z3.shape, output.shape)
#print(computCost(params, input_size, hidden_size, num_labels, X, y_oneshot, learning_rate))
J, grad = back_propagate(params, input_size, hidden_size, num_labels, X, y_oneshot, learning_rate)
print(J, grad.shape)

fmin = minimize(fun = back_propagate, x0 = params, args = (input_size, hidden_size, num_labels, X, y_oneshot, learning_rate)
, method = 'TNC', jac = True, options = {'maxiter':250})        #最多訓練250次
print(fmin)        #輸出x為theta1 ,theta2 optimal

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))