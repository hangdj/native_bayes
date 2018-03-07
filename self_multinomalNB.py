# coding:utf-8
import numpy as np
#以单词为计量单位
class MultinomialNB(object):
    '''
    :parameter
        alpha :平滑参数
            -->0 不平滑
            -->0-1 Lidstone 平滑
            -->1 Laplace 平滑
        fit_prior:boolean
            是否学习类先验概率。
      如果fasle，则会使用统一的优先级。
        class_prior:array-like, size (n_classes,) 数组格式 大小 类别个数
            这些类的先验概率，如果指定的话，先验概率将不会根据数据计算
    Attributes

    fit(X,y):特征和标签 都是数组

    predict(X:
    '''
    def __init__(self,alpha=1.0,fit_prior=True,class_prior=None):
        self.alpha=alpha
        self.fit_prior=fit_prior
        self.class_prior=class_prior
        self.classes=None
        self.conditional_prob=None
    def _calculate_feature_prob(self,feature):#计算条件概率
        values=np.unique(feature)
        #print(values,'sddd')
        total_num=float(len(feature))
        value_prob={}
        for v in values:#有平滑效果
            value_prob[v]=((np.sum(np.equal(feature,v))+self.alpha)/(total_num+len(values)*self.alpha))
        return value_prob
    def fit(self,X,y):
        self.classes=np.unique(y)
        #计算先验概率
        if self.class_prior==None:#不指定先验概率
            class_num=len(self.classes)#类别个数
            if not self.fit_prior:#是否自学先验概率，false则统一指定
                self.class_prior=[1.0/class_num for _ in range(class_num)]#统一优先级
            else:
                self.class_prior=[]
                sample_num=float(len(y))
                for c in self.classes:
                    c_num=np.sum(np.equal(y,c))
                    self.class_prior.append((c_num+self.alpha)/(sample_num+class_num*self.alpha))#根据标签出现个数计算优先级
                #print(self.class_prior,"class_prior")
        #计算条件概率
        self.conditional_prob={}#like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{}, c1:{...} }
        for c in self.classes:
            self.conditional_prob[c]={}
            #print(len(X[0]),"X[0]")
            for i in range(len(X[0])):#特征总数2
                #print(X[np.equal(y,c)][:,1],"y==c")
                feature=X[np.equal(y,c)][:,i]#这里加个逗号才是遍历所有行！！！
                #print(feature,i,"feature")
                self.conditional_prob[c][i]=self._calculate_feature_prob(feature)
        print(self.conditional_prob,"conditional_prob")
        return self

    #给了 单词概率{value0:0.2,value1:0.1,value3:0.3,.. } 和目标值 给出目标值的概率
    def _get_xj_prob(self, values_prob,target_value):
         return values_prob.get(target_value)
    #依据(class_prior,conditional_prob)预测一个简单地样本
    def _predict_single_sample(self,x):
        label=-1
        max_posterior_prob=0
        #对每一个类别，计算其后验概率：class_prior*conditional_prob
        for c_index in range(len(self.classes)):
            current_class_prior=self.class_prior[c_index]#类别优先级，类别多的优先级大
            current_conditional_prob=1.0#条件概率
            feature_prob=self.conditional_prob[self.classes[c_index]]#类别1的条件概率
            #print(feature_prob.keys(),"feature_prob")
            j=0
            for feature_i in feature_prob.keys():#每一个特征下的概率
                current_conditional_prob*=self._get_xj_prob(feature_prob[feature_i],x[j])
                j+=1

            #比较后验概率，更新max_posterior_prob ,label
            print( current_class_prior*current_conditional_prob,self.classes[c_index],"后验概率")
            if current_class_prior*current_conditional_prob>max_posterior_prob:#取最大的后验概率
                max_posterior_prob=current_class_prior*current_conditional_prob
                label=self.classes[c_index]

        return label
    def predict(self,X):
        if X.ndim==1:
            return self._predict_single_sample(X)
        else:
            labels=[]
            for i in range(X.shape(0)):
                label=self._predict_single_sample(X[i])
                labels.append(label)
            return labels
if __name__=='__main__':
    X = np.array([
                          [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                          [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
                 ])
    X = X.T
    y = np.array(        [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    nb = MultinomialNB(alpha=1.0,fit_prior=True)
    nb.fit(X,y)
    print ("[2,5]-->",nb.predict(np.array([2,5])))#输出1 0.08169 0.04575
    print ("[2,4]-->",nb.predict(np.array([2,4])))#输出-1 0.0327 0.0610