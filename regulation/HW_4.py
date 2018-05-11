# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:10:28 2018

@author: user1
"""
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def predict(X, theta) :
    prob = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in prob]      #串列生成

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))  #np.exp() Calculate the exponential of all elements in the input array

def costReg(theta, X, Y, lambda1, reg_on = True):
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)
    seg1 = np.multiply(Y, np.log(sigmoid(X * theta.T)))    #elementwise product
    seg2 = np.multiply((1 - Y), np.log(1 - sigmoid(X * theta.T)))
    if(reg_on == True) :
        reg = (lambda1 /(2 * len(X))) * np.sum(np.power(theta, 2))
        return np.sum(- (seg1 + seg2)) / len(X) + reg
    else :
        return np.sum(- (seg1 + seg2)) / len(X)

def gradientReg(theta, X, Y, lambda1):     #算gradient
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)
    parameters = int(theta.ravel().shape[1])
    gradient = np.zeros(parameters)         #用來存新的theta
    error = sigmoid(X * theta.T) - Y
    
    for i in range(parameters) :
        term = np.multiply(error, X[:, i])
        if(i == 0) :
            gradient[i] = sum(term) / len(X)
        else :
            gradient[i] =  sum(term) / len(X) + (lambda1 / len(X)) * theta[:, i]
    return gradient
    
    

path = os.getcwd() + '\exercise4-data\ex2data2.txt'
data = pd.read_csv(path, header = None, names =['Test 1', 'Test 2', 'Accepted'])
#print(data)

positive = data[data['Accepted'].isin([1])]  #查看共有多少產品錄取
negative = data[data['Accepted'].isin([0])]
#劃出點圖，s代表點的scalar
#fig1, ax1 = plt.subplots(figsize=(12, 8))
#ax1.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')  #藍色代表錄取
#ax1.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected') #紅色代表不錄取
#ax1.set_xlabel('Test 1 Score')
#ax1.set_ylabel('Test 2 Score')

degree = 6
x1 = data['Test 1']
x2 = data['Test 2']
data.insert(3 ,'Ones', 1)      #加入x0項

for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data['Feat.' + str(i - j) + str(j)] = np.power(data['Test 1'], i - j) * np.power(data['Test 2'], j)

data.drop('Test 1', axis = 1, inplace = True)
data.drop('Test 2', axis = 1, inplace = True)

cols = data.shape[1]
X = data.iloc[:, 1 : cols]    #取column 1 ~ 28
Y = data.iloc[:, 0 : 1]

theta = np.zeros(cols - 1)
X = np.array(X.values)
Y = np.array(Y.values)
lambda1 = 1
#print(costReg(theta, X, Y, lambda1))
#print(gradientReg(theta, X, Y, lambda1))


data = data.sample(frac = 1)
X2 = data.iloc[:, 1 : cols]    #取column 1 ~ 28
Y2 = data.iloc[:, 0 : 1]
train_num = int(data.shape[0] * 0.7)
val_num = int(data.shape[0] * 0.2)
X_train = X2.iloc[: train_num, :]        #用來做訓練
Y_train = Y2.iloc[: train_num, :]
X_valid = X2.iloc[train_num : train_num + val_num, :]   #用來測量效用
Y_valid = Y2.iloc[train_num : train_num + val_num, :]
X_test = X2.iloc[train_num + val_num :, :]              #用來測試真實情況
Y_test = Y2.iloc[train_num + val_num :, :]


lambdaArray = np.arange(0, 10, 1)
cost = np.zeros(lambdaArray.shape[0])
theta_all = list()

for i in range(lambdaArray.shape[0]):
    theta = np.zeros(X_train.shape[1])
    result = opt.fmin_tnc(func = costReg, x0 = theta, fprime = gradientReg, args=(X_train, Y_train, lambdaArray[i]), )
    theta_min = np.matrix(result[0])
    cost[i] = costReg(theta_min, X_valid, Y_valid, lambdaArray[i], reg_on = False)     #不需要再做regulation的cost運算
    theta_all.append(theta_min.ravel())

fig, ax = plt.subplots(figsize = (9, 8))
ax.set_xlabel("Lambda")
ax.set_ylabel("Cost")
ax.plot(lambdaArray, cost)
index_min = cost.argmin()
lambda1 = lambdaArray[index_min]

result = opt.fmin_tnc(func = costReg, x0 = theta, fprime = gradientReg, args=(X, Y, 1), )   #theta要的是array，不然會有error
print(result)
theta_min = np.matrix(result[0])
predictions = predict(X, theta_min)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct)) * 100
print("accuracy with training data and lambda: %f = %f" %(1, accuracy))

#################################畫圖############################

h = 0.02
xx, yy = np.meshgrid(np.arange(-1, 1.5, h), np.arange(-1, 1.5, h)) #Return coordinate matrices from coordinate vectors
X2_plot = np.ones(xx.ravel().shape[0]).ravel()        #Return a new array of given shape and type, filled with ones.
                                                   #X2_plot用來儲存decision boundary
for i in range(1, degree + 1):
    for j in range(0, i + 1): 
        term = np.power(xx.ravel(), i - j) * np.power(yy.ravel(), j)
        X2_plot = np.c_[X2_plot, term.ravel()]  #Translates slice objects to concatenation along the second axis.

Z = np.matrix(predict(np.matrix(X2_plot), theta_min)).reshape(xx.shape)
fig2, ax2 = plt.subplots(figsize = (12, 9))
ax2.scatter(positive['Test 1'], positive['Test 2'], s = 50, c = 'b', marker = 'o', label = "Accepted")
ax2.scatter(negative['Test 1'], negative['Test 2'], s = 50, c = 'r', marker = 'x', label = "Rejected")
ax2.contour(xx, yy, Z, cmap = plt.cm.Paired)
ax2.legend()          #加上右上角的標註
ax2.set_xlabel("Test1 Score")
ax2.set_ylabel("Test2 Score")


theta_optimun = theta_all[index_min]
X_test = np.matrix(X_test)
predictions = predict(X_test, theta_optimun)
Y_test = np.array(Y_test)
correct2 = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y_test)]
accuracy = (sum(map(int, correct2)) / len(correct2)) * 100
print("Real accuracy with lambda: %f = %f"%(lambda1 , accuracy))

#################################畫圖############################

h = 0.02
xx, yy = np.meshgrid(np.arange(-1, 1.5, h), np.arange(-1, 1.5, h)) #Return coordinate matrices from coordinate vectors
X3_plot = np.ones(xx.ravel().shape[0]).ravel()        #Return a new array of given shape and type, filled with ones.
                                                   #X2_plot用來儲存decision boundary
for i in range(1, degree + 1):
    for j in range(0, i + 1): 
        term = np.power(xx.ravel(), i - j) * np.power(yy.ravel(), j)
        X3_plot = np.c_[X3_plot, term.ravel()]  #Translates slice objects to concatenation along the second axis.

Z2 = np.matrix(predict(np.matrix(X3_plot), theta_optimun)).reshape(xx.shape)
fig3, ax3 = plt.subplots(figsize = (12, 9))
ax3.scatter(positive['Test 1'], positive['Test 2'], s = 50, c = 'b', marker = 'o', label = "Accepted")
ax3.scatter(negative['Test 1'], negative['Test 2'], s = 50, c = 'r', marker = 'x', label = "Rejected")
ax3.contour(xx, yy, Z2, cmap = plt.cm.Paired)
ax3.legend()          #加上右上角的標註
ax3.set_xlabel("Test1 Score")
ax3.set_ylabel("Test2 Score")






