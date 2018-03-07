#coding:utf-8
import numpy as np
from sklearn.naive_bayes import MultinomialNB,GaussianNB
'''
X=np.random.randint(5,size=(6,100))
y=np.array([1,2,3,4,5,6])
#print(X,y)
clf=MultinomialNB()
clf.fit(X,y)
MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
print(clf.predict([X[5]]))
--------------------
#多项式模型在训练一个数据集结束之后可以继续训练其他数据集而无需将两个数据集放在一起训练。
# partial_fit()可以进行这样的训练，适合无法一次将训练集加入到内存
clf=MultinomialNB()
clf.partial_fit([np.array([1,1])],np.array(['aa']),['aa','bb'])
GaussianNB()
clf.partial_fit([np.array([6,1])],np.array(['bb']))
GaussianNB()
clf.predict([[9,1]])'''

X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
             ])
X = X.T
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

nb = MultinomialNB(alpha=1.0,fit_prior=True)
nb.fit(X,y)
print (nb.predict(np.array([[1,4]])))#输出-1