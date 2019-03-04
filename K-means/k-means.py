import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs


class K_mean(object):
    
    def __init__(self,c,n_iterations):
        self.n_iterations=n_iterations
        self.c=c
    
    def init_params(self,X):
        self.X=X
        self.init_u()
    
    def init_u(self):
        all_u=[]
        for i in range(self.c):
            uid=np.random.choice(self.X.shape[0],1)
            u=self.X[uid]
            delta=u-self.X
            d_delta=np.apply_along_axis(lambda x:np.dot(x,x),1,delta)
            best_u=X[np.argmax(d_delta)]
            all_u.append(best_u)
        self.U=np.array(all_u)
        
    
    def get_label(self):
        labels=[]
        for i in range(len(self.X)):
            di=self.X[i].reshape(1,-1)-self.U
            di=np.apply_along_axis(lambda x:x.dot(x),1,di)
            k=di.argmin()
            labels.append(k)
        return labels
    
    def update_u(self):
        U_up=[]
        labels=self.get_label()
        X_new=np.c_[self.X,labels]
        self.X_complete=X_new
        label_map=set(labels)
        for i in label_map:
            group=X[labels==i]
            u=group.sum(axis=0)/len(group)
            U_up.append(u)
        self.U=np.array(U_up)
        
    def fit_transform(self,X):
        self.init_params(X)
        old_u=None
        iters=0
        for iter in range(self.n_iterations):
            self.update_u()
            iters+=1
        self.iters=iters
        
        return self.X_complete


# 绘图函数
def plot_img(x,y=True):
    fig=plt.figure(figsize=(7,5))
    if y:
        plt.scatter(x[:,0],x[:,1],c=x[:,2],cmap=plt.cm.Spectral,marker='o')
    else:
        plt.scatter(x[:,0],x[:,1],marker='o')
    plt.title('Scatter')
    plt.xlabel('feature 2')
    plt.ylabel('feature 1')
    plt.show()


'''构造数据'''
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，
#簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X,Y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], 
                  random_state =9)

c=4
n_iters=100
km=K_mean(c,n_iters)
X_show=km.fit_transform(X)
plot_img(X_show)