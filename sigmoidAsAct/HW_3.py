# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:28:30 2018

@author: user1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#func to compute the total cost error
def computeCost(X, y, theta) :    #X represents training sets
    inner = np.power(((X * theta.T) - y), 2)   #inner is also a matrix object (brocasting)
    return np.sum(inner) / (2 * X.shape[0])

def gradientDescent(theta, X, y, alpha, iters) :
    temp = np.matrix(np.zeros(theta.shape))  #numpy.zeros會回傳一個限定維度的array object
    parameters = int(theta.ravel().shape[1])  #matrix.ravel()會回傳個整個攤平的matrix(即row vector)
    cost = np.zeros(iters)  
    current = 0
    for i in range(iters):
        error = (X * theta.T) - y       #error為97 x 1
        for j in range(parameters):      #parameters即為θ的數量, 進行θ0, θ1的運算  
            term = np.multiply(error, X[:,j])    #矩陣乘法, X[:, j]為 97 x 2矩陣的column 0, 1 (bitwise product)
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))   #temp為 1 x 2 matrix
            
        term1 = np.multiply(error, X[:,0])
        temp[0,0] = theta[0,0] - ((alpha / len(X)) * np.sum(term1))
        term2 = np.multiply(error, X[:,1])
        temp[0,1] = theta[0,1] - ((alpha / len(X)) * np.sum(term2)) 
        term3 = np.multiply(error, X[:,2])
        temp[0,1] = theta[0,2] - ((alpha / len(X)) * np.sum(term3))         
        
        theta = temp
        cost[i] = computeCost(X, y, theta)
        current += 1
        if (cost[i - 1] - cost[i] < 10**-10) & (i > 0):
            print(cost[i - 1])
            print(cost[i])
            print("Early Stop at %d iters" % current)
            break
        
    print("θ0: %f θ1: %f" %(theta[0, 0], theta[0, 1]))    
    return theta, cost    

path = os.getcwd() + '\ex1data2.txt'
data = pd.read_csv(path, header = None, names =['Size', 'Bedrooms', 'Price'])
#print(data)

data = (data - data.mean()) / data.std()  #DataFrame return standard deviation
mean = data.mean()       #mean is DataFrame Object
standardDeviation = data.std()
#print(data)

data.insert(0, 'Ones', 1)     #插入column insert(loc, column, value, allow_duplicates=False)
columns = data.shape[1]       #data現在為100 X 4
print(data)
x = data.iloc[:, 0:(columns -1)]           #利用slice取出處理後的DataFrame
y = data.iloc[:, (columns - 1):columns]
X = np.matrix(x.values)
Y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0]))

alpha = 0.0001
iters = 10000

g, cost = gradientDescent(theta, X, Y, alpha, iters)

fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')




