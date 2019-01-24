import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 将数据进行z-score标准化
df=pd.read_csv('LogiReg_data.txt',names=['feature_1','feature_2','class'])
ss=StandardScaler()
x=ss.fit_transform(df[['feature_1','feature_2']])
y=df['class'].values

# 逻辑回归算法
class LogisticRegression(object):
    
    # sigmoid函数
    def sigmoid(self,x):
        theta=LogisticRegression.theta
        return 1/(1+np.exp(-x.dot(theta)))
    
    # 决策函数
    def h(self,X):
        return np.apply_along_axis(self.sigmoid,1,X)
    
    # 梯度下降法
    def gd_fit(self,X,y,n,a):
        X=np.c_[X,np.ones((len(x),1))]
        LogisticRegression.theta=np.ones((len(X[0]),1)).ravel() # 初始化参数
        for i in range(n):
            grad=X.T.dot(self.h(X).ravel()-y) # 构造梯度
            theta=LogisticRegression.theta-a*grad 
        LogisticRegression.theta=theta
        
    # 牛顿法
    def newton_fit(self,X,y,n):
        X=np.c_[X,np.ones((len(x),1))]
        LogisticRegression.theta=np.ones((len(X[0]),1)).ravel()
        for i in range(n):
            try:
                # 构造hessian矩阵
                hessian=X.T.dot(np.diag(self.h(X).ravel())).dot(np.diag(1-self.h(X).ravel())).dot(X) 
                grad=X.T.dot(self.h(X).ravel()-y) # 构造梯度
                step=np.linalg.inv(hessian).dot(grad)  
                theta=LogisticRegression.theta-step
            except:
                break # 防止hession矩阵奇异 
        LogisticRegression.theta=theta
    
    # 计算预测值
    def predict(self,X,threshold):
        LogisticRegression.threshold=threshold
        X=np.c_[X,np.ones((len(X),1))]
        y=np.where(self.h(X).ravel()>threshold,1,0)
        return y
    
    # 计算样本为正类的概率
    def predict_proba(self,X):
        X=np.c_[X,np.ones((len(X),1))]
        return self.h(X).ravel()
   
    # 计算精度
    def score(self,y_pred,y_true):
        y_pred_true=y_pred[y_pred==y_true]
        tp=len(y_pred_true[y_pred_true==1])
        return tp/(len(y_pred[y_pred==1]))
    
    # 计算AUC值
    def auc(self,proba):
        proba=np.sort(proba)
        threshold=LogisticRegression.threshold
        m=proba[np.where(proba>threshold)].shape[0]
        n=proba[np.where(proba<threshold)].shape[0]
        down=m*n
        up=np.where(proba>threshold)[0].sum()-m*(m+1)/2
        return up/down
       
# 初始化参数
n=1000
a=0.1
threshold=0.5

# 构造模型，并训练
lr=LogisticRegression()
#lr.gd_fit(x,y,n,a)
lr.newton_fit(x,y,n)

# 计算theta和预测值
theta=lr.theta
print('theta：',theta)
y_pred=lr.predict(x,threshold)

# 计算模型精度
score=lr.score(y_pred,y)
print('精度：',score)

# 得到预测样本为正类的概率
proba=lr.predict_proba(x)

# 计算AUC值
auc=lr.auc(proba)
print('AUC:',auc)

# 数据可视化

# 拓展一个含有500*500个样本点的空间
N=500
M=500

f1_min,f2_min=df[['feature_1','feature_2']].min()
f1_max,f2_max=df[['feature_1','feature_2']].max()

f1=np.linspace(f1_min,f1_max,N)
f2=np.linspace(f2_min,f2_max,M)

# 创建样本集，并进行预测
m1,m2=np.meshgrid(f1,f2) # 建立两个坐标矩阵

x_show=np.stack((m1.flat,m2.flat),axis=1) # 创建样本集
x_show=ss.fit_transform(x_show) #标准化样本集
y_show=lr.predict(x_show,threshold)

# 可视化
sns.set()

# 分类图
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
plt.pcolormesh(m1,m2,y_show.reshape(m1.shape),cmap=cm_light)

# 散点图
plt.scatter(df.loc[df['class']==1,'feature_1'],df.loc[df['class']==1,'feature_2'],label='1')
plt.scatter(df.loc[df['class']==0,'feature_1'],df.loc[df['class']==0,'feature_2'],c='coral',label='0')

plt.title('Scatter')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.legend()
plt.show()