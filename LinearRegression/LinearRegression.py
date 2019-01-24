import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np
import seaborn as sns

# 线性回归
class LinearRegression(object):
    
    # 正规方程 
    def ne_fit(self,X,y):
        X=np.c_[X,np.ones((X.shape[0],1))] # 将截距项加入特征矩阵
        X_T=X.T # 转置矩阵
        X_inv=np.linalg.inv(X_T.dot(X)) # 构造逆矩阵
        theta=X_inv.dot(X_T).dot(y) # 参数及截距矩阵
        LinearRegression.params= theta[:-1]
        LinearRegression.intercept=theta[-1]
        LinearRegression.theta=theta
        
    # 梯度下降
    def gd_fit(self,X,y,alpha,maxites):
        X=np.c_[X,np.ones((X.shape[0],1))]
        theta=np.ones((X.shape[1],1)) # 初始化参数值
        X_T=X.T
        for i in range(0,maxites):
            loss=X.dot(theta)-y # 损失函数
            gradient=X_T.dot(loss)/X.shape[0] # 梯度
            theta=theta-alpha*gradient # 迭代theta
        LinearRegression.params= theta[:-1]
        LinearRegression.intercept=theta[-1]
        LinearRegression.theta=theta
        
    def predict(self,X):
        X=np.c_[X,np.ones((X.shape[0],1))]
        return X.dot(self.theta)

# 加载内置波士顿房价数据集
boston=load_boston()
X=pd.DataFrame(boston.data,columns=boston.feature_names)
X=pd.DataFrame(X['RM'].head(100))
y=pd.DataFrame(boston.target).head(100)
x_train=X.values
y_train=y.values
print('x_train: ',x_train.shape)
print('y_train:',y_train.shape)

x_test=np.arange(x_train.min(),x_train.max().max(),0.01)

# 正规方程
LR_ne=LinearRegression()
LR_ne=LinearRegression()
LR_ne.ne_fit(x_train,y_train)
y_pred_ne=LR_ne.predict(x_test)
print('正规方程：')
print('参数:',LR_ne.params)
print('截距:',LR_ne.intercept.shape,end='\n\n')

# 梯度下降
LR_gd=LinearRegression()
a=0.1
count=100
LR_gd.gd_fit(x_train,y_train,alpha=0.01,maxites=30000)
y_pred_gd=LR_gd.predict(x_test)
print('梯度下降：')
print('参数:',LR_gd.params)
print('截距:',LR_gd.intercept.shape)

# 每栋住宅的房间数与房价的关系图
sns.set()
fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(221)
ax1.scatter(X,y)
plt.title('House prices in Boston')
plt.xlabel('number of rooms')
plt.ylabel('price')

# 用一条直线来拟合数据
ax2=fig.add_subplot(222)
plt.scatter(X,y)                      
plt.plot(x_test,y_pred_ne,label='normal equation')
plt.plot(x_test,y_pred_gd,label='gradient descent')
plt.title('House prices in Boston')
plt.xlabel('number of rooms')
plt.ylabel('price')
plt.legend()
plt.show()