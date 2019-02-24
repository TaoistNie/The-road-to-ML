import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import pandas as pd

class SVM(object):

	def __init__(self,max_iterations,kernal='linear'):
		
		self.max_iterations=max_iterations
		self._kernal=kernal

	def _init_params(self,X,Y):

 		self.m=X.shape[0]
 		self.n=X.shape[1]
 		self.X=X
 		self.Y=Y
 		self.b=0.0
 		self.alpha=np.ones(self.m)
 		self.E=[ self._E(i) for i in range(self.m)]
 		self.C=1.0
 		

	def _k(self,xi,xj):
		if self._kernal=='linear':
 			return sum([xi[i]*xj[i] for i in range(self.n)])

	def _g(self,i):
		
		g=0
		for j in range(self.m):
			g+=self.alpha[j]*self.Y[j]*self._k(self.X[j],self.X[i])+self.b 
		return g

	def _E(self,i):

 		E=self._g(i)-self.Y[i]

 		return E

	def _KKT(self,i):
		y_g=self.Y[i]*self._g(i)
		if self.alpha[i]==0:
 			return y_g>1
		elif self.alpha[i]>0 and self.alpha[i]<self.C:
 			return y_g==1
		else:
 			return y_g<1

	def _init_alpha(self):

		index_list=[i for i in range(self.m) if 0<self.alpha[i]<self.C]
		non_index_list=[i for i in range(self.m) if i not in index_list]
		index_list.extend(non_index_list)

		for i in index_list:
			if self._KKT(i):
				continue
			E1=self.E[i]

			if E1>=0:
				j=min(range(self.m),key=lambda x:self.E[x])
			else:
				j=max(range(self.m),key=lambda x:self.E[x])
			return (i,j)

	def _compare(self,alpha2,L,H):
		if alpha2>H:
			return H
		elif alpha2<L:
			return L
		else:
			return alpha2

	def fit(self,features,labels):

		self._init_params(features,labels)

		for iter in range(self.max_iterations):

			i,j=self._init_alpha()

			if self.Y[i]!=self.Y[j]:
				L=max(0,self.alpha[j]-self.alpha[i])
				H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
			elif self.Y[i]==self.Y[j]:
				L=max(0,self.alpha[j]+self.alpha[i]-self.C)
				H=min(self.C,self.alpha[j]+self.alpha[i])
			
			E1=self.E[i]
			E2=self.E[j]

			eta=self._k(self.X[i],self.X[i])+self._k(self.X[j],self.X[j])-2*self._k(self.X[i],self.X[j])
			if eta<=0:
				continue
			
			alpha2_new_unc=self.alpha[j]+self.Y[j]*(E2-E1)/eta
			alpha2_new=self._compare(alpha2_new_unc,L,H)
			
			alpha1_new=self.alpha[i]+self.Y[i]*self.Y[j]*(self.alpha[j]-alpha2_new)

			b1=-E1 - self.Y[i]*self._k(self.X[i],self.X[i])*(alpha1_new-self.alpha[i])-self.Y[j]*self._k(self.X[j],self.X[i])*(alpha2_new-self.alpha[j])+self.b
	
			b2=-E2 - self.Y[i]*self._k(self.X[i],self.X[j])*(alpha1_new-self.alpha[i])-self.Y[j]*self._k(self.X[j],self.X[j])*(alpha2_new-self.alpha[j])+self.b

			if alpha1_new>0 and alpha1_new<self.C:
				b_new=b1

			elif alpha2_new>0 and alpha2_new<self.C:
				b_new=b2
			else:
				b_new=(b1+b2)/2

			self.alpha[i]=alpha1_new
			self.alpha[j]=alpha2_new
			self.b=b_new

			self.E[i]=self.E[i]
			self.E[j]=self.E[j]

		return 'train done!'

	def predict(self,data):
		result=0
		for i in range(self.m):
			result+=self.alpha[i]*self.Y[i]*self._k(self.X[i],data)+self.b
		return 1 if result>0 else -1
		
	def score(self,data,labels):
		Y_pred=[]
		for i in range(len(data)):
			y_pred=self.predict(data[i])
			Y_pred.append(y_pred)
		Y_pred=np.array(Y_pred)
		score=1-len(labels[labels+Y_pred==0])/len(labels)
		return score


# 构造数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

max_iterations=200
svm=SVM(max_iterations)
svm.fit(X_train,y_train)
score=svm.score(X_test,y_test)
print(score)