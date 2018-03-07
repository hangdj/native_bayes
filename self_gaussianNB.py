# coding:utf-8
#当特征是连续变量的时候，运用多项式模型就会导致很多P(xi|yk)==0（不做平滑的情况下），
# 此时即使做平滑，所得到的条件概率也难以描述真实情况。所以处理连续的特征变量，应该采用高斯模型。

#高斯模型假设每一维特征都服从高斯分布,需要计算每一维的均值和方差
from self_multinomalNB import MultinomialNB
import numpy as np
#继承Multinomial并重载相应的方法
class GaussianNB(MultinomialNB):
    def _calculate_feature_prob(self,feature):#计算平均值和方差
        mu=np.mean(feature)
        sigma=np.std(feature)
        return (mu,sigma)
    #计算高斯分布的概率密度
    def _prob_gaussian(self,mu,sigma,x):
        return (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
    def _get_xj_prob(self, mu_sigma,target_value):
        return self._prob_gaussian(mu_sigma[0],mu_sigma[1],target_value)
if __name__=='__main__':
    X = np.array([
                          [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                          [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
                 ])
    X = X.T
    y = np.array(        [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    nb = GaussianNB(alpha=1.0,fit_prior=True)
    nb.fit(X,y)
    print ("[2,5]-->",nb.predict(np.array([2,5])))#输出1 0.08169 0.04575
    print ("[2,4]-->",nb.predict(np.array([2,4])))#输出-1 0.0327 0.0610