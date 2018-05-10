# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:19:17 2018

@author: Lin
"""
#import the packages
import numpy as np
import matplotlib.pyplot as plt
#畫出原f(theta)方程式，從-5到5之間，間距為0.0001
theta = np.arange(-5.0, 5.0, 10**-4)
func = theta**4 - 3*theta**3 + 2
plt.plot(theta, func)
plt.show()
#產生隨機的初始位置
currentPosition = np.random.uniform(high = 5.0, low = -5.0, size = None)
currentPosition = 0.04
initialPosition = currentPosition
epsilon = 10**-10   #設定最小優化區間
lambdA = 10**-5    #設定lambda大小
ValueOfFunction = currentPosition**4 - 3*currentPosition**3 + 2 #利用f'(theta)去尋找某範圍中的最低點(saddle)
trackOfValue = [ValueOfFunction]                                #並利用Tuple紀錄函數值
finding = False
#利用whhile迴圈去不斷地逼近saddle，並到了一定的位置後會自動停止，達到early stop的效果
while not finding:
    #利用 x2 = x1 - f(theta) * lambda的公式去逼近saddle 
    currentPosition = currentPosition - lambdA * (4 * currentPosition ** 3 - 9 * currentPosition ** 2)
    ValueOfFunction = currentPosition ** 4 - 3 * currentPosition ** 3 + 2
    trackOfValue.append(ValueOfFunction) #函數值加到Tuple中
    #print("f\'(theta) : %s" %((4 * currentPosition ** 3 - 9 * currentPosition ** 2)))
    #print("Now at %s" %currentPosition)
    if trackOfValue[-2] - trackOfValue[-1] < epsilon:   #當位置與位置間的差距比epsilon還小時就停止逼近
        print("Early Stop")
        finding = True
        
def findGlobalMinimum(left, right, trackOfValues) : #trackOfValues用來儲存所有local minimum，最後只要在其中找到最小值就好
    if((right - left) > 0.0001) :                   #right, left個代表區間的上下限
        #分成兩半去進行同個function
        findGlobalMinimum(left, (left + right) / 2, trackOfValues)
        findGlobalMinimum((left + right) / 2, right, trackOfValues)
    else :
        current = np.random.uniform(high = right, low = left, size = None)
        epsilon = 10**-10   
        lambdA = 10**-5    
        Value = current**4 - 3*current**3 + 2 #利用f'(theta)去尋找某範圍中的最低點(saddle)
        trackOfValues.append(Value)
        finding = False
        while not finding:
            #利用 x2 = x1 - f(theta) * lambda的公式去逼近saddle 
            current = current- lambdA * (4 * current ** 3 - 9 * current ** 2)
            ValueOfFunc = current ** 4 - 3 * current ** 3 + 2
            trackOfValues.append(ValueOfFunc) #函數值加到list中
            if trackOfValues[-2] - trackOfValues[-1] < epsilon:   #當位置與位置間的差距比epsilon還小時就停止逼近
                print("Early Stop")
                finding = True          
    
    
 
print("f\'(theta) : %s" %((4 * currentPosition ** 3 - 9 * currentPosition ** 2)))       
print("Start at %s" %initialPosition)
print("End at %s" %currentPosition)
print("With %s approaching" %len(trackOfValue)) 
print("The local optimal is 2.25")       
#畫出函數值的變化            
plt.plot(np.arange(len(trackOfValue)),trackOfValue)
plt.show()
        
    

