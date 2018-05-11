import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

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
        #for j in range(parameters):      #parameters即為θ的數量, 進行θ0, θ1的運算  
            #term = np.multiply(error, X[:,j])    #矩陣乘法, X[:, j]為 97 x 2矩陣的column 0, 1 (bitwise product)
            #temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))   #temp為 1 x 2 matrix
            
        term1 = np.multiply(error, X[:,0])
        temp[0,0] = theta[0,0] - ((alpha / len(X)) * np.sum(term1))
        term2 = np.multiply(error, X[:,1])
        temp[0,1] = theta[0,1] - ((alpha / len(X)) * np.sum(term2))        
        
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

path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
data = data.append({'Population': 98, 'Profit': -60}, ignore_index = True)
data = data.append({'Population': 80, 'Profit': 57}, ignore_index = True)
data = data.append({'Population': 94, 'Profit': -58}, ignore_index = True)
#data = data.append({'Population': 194, 'Profit': -58}, ignore_index = True)
data.head()
data.describe()

#data.plot(kind='scatter', x='Population', y='Profit', figsize=(10, 5))

data.insert(0, 'Ones', 1)

cols = data.shape[1]      #shape is a tuple attribute 代表data維度, shape[1]查看第二維的維度

X = data.iloc[:,0:cols-1]   
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)             #X, y變成matrix Objects
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0])) #theta 也為1X2維矩陣，值都為零 , (0, 0)為起點

alpha = 0.0001
iters = 100000

g, cost = gradientDescent(theta, X, y, alpha, iters)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(np.arange(iters), cost, 'r')
#ax.set_xlabel('Iterations')
#ax.set_ylabel('Cost')
#ax.set_title('Error vs. Training Epoch')


model2 = linear_model.LinearRegression(normalize = True)
model2.fit(X, y)


x = np.array(X[:, 1].A1)
f = model2.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')





