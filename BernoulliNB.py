#coding:utf-8
import numpy as np
from sklearn.naive_bayes import BernoulliNB
X=np.random.randint(2,size=(6,100))
Y=np.array([1,2,3,4,4,5])
clf=BernoulliNB()
clf.fit(X,Y)
BernoulliNB(alpha=1.0,binarize=0.0,class_prior=None,fit_prior=True)
print(clf.predict([X[4]]))