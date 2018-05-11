# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:48:55 2018

@author: user1
"""
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#define sigmoid func
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))  #np.exp() Calculate the exponential of all elements in the input array

#define a func to compute the cost
def computeCost(theta, X, Y) :
    theta = np.matrix(theta)
    s1 = np.multiply(Y, np.log(sigmoid(X * theta.T)))    #elementwise product
    s2 = np.multiply((1 - Y), np.log(1 - sigmoid(X * theta.T)))
    return -(1 / len(X)) * np.sum(s1 + s2)

#define a func to compute gradient
def computeGradient(theta, X, Y) :
    theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1])
    gradient = np.zeros(theta.T.shape[0])
    error = sigmoid(X * theta.T) - Y
    
    for i in range(parameters) :
        term = np.multiply(error, X[:,i])
        gradient[i] = np.sum(term) / len(X)
        
    return gradient

#define a func to use our model to predict the result
def predict(X, theta) :
    prob = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in prob]
        
path = os.getcwd() + '\data\ex2data1.txt'
data = pd.read_csv(path, header = None, names =['Exam 1', 'Exam 2', 'Admitted'])
#print(data)

positive = data[data['Admitted'].isin([1])]  #查看共有多少人錄取
negative = data[data['Admitted'].isin([0])]
#劃出點圖，s代表點的scalar
#fig1, ax1 = plt.subplots(figsize=(12, 8))
#ax1.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  #藍色代表錄取
#ax1.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted') #紅色代表不錄取
#ax1.set_xlabel('Exam 1 Score')
#ax1.set_ylabel('Exam 2 Score')

#test the sigmoid func
#nums = np.arange(-10, 10, 1)
#fig2, ax2 = plt.subplots(figsize = (12, 8))
#ax2.plot(nums, sigmoid(nums), 'b')

data.insert(0, 'Ones', 1)     #插入column insert(loc, column, value, allow_duplicates=False)
data.insert(4, 'Square_1', data['Exam 1'] **4 )
#data.insert(5, 'Square_2', data['Exam 2'] **3 )
print(data)
columns = data.shape[1]       #data現在為100 X 5


x = data.iloc[:, [0, 1, 2, 4]]
y = data.iloc[:, 3:4]
#x = data.iloc[:, 0:(columns -1)]           #利用slice取出處理後的DataFrame
#y = data.iloc[:, (columns - 1):columns]
X = np.matrix(x.values)
Y = np.matrix(y.values)
theta = np.zeros(4)



result = opt.fmin_tnc(func = computeCost, x0 = theta, fprime = computeGradient, args=(X, Y))   #theta要的是array，不然會有error
#print(result)       #result為一個array包含theta, iters
#print(computeCost(result[0], X, Y))

#==============================test our model===================================

theta_min = np.matrix(result[0])
predictions = predict(X, theta_min)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) % len(correct))
print("accuracy = %f" %accuracy)
print("iters : %d" %result[1])

h = 0.02 # step size in the mesh
#create a mesh to plot in
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = predict(np.c_[np.ones(xx.ravel().shape[0]).ravel(), xx.ravel(), yy.ravel(), (xx**4).ravel()], theta_min)
Z = np.matrix(Z).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')