# coding :utf-8
from self_multinomalNB import MultinomialNB
#以文档为计量单位
import numpy as np
class BernoulliNB(object):
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

    def createVocabList(self,dataSet):  # 统计词汇，创建词典
        vocabSet = set([])
        for document in dataSet:
            #print(document)
            vocabSet = vocabSet | set(document)
            # print (vocabSet)
        return list(vocabSet)

    def setOfWord2Vec(self,vocabList, inputSet):  # 创建词频统计向量,此方法为词集模型，出现赋值1，or0
        returnVec = []  # 返回的文本向量与词典大小保持一致
        dims = inputSet.shape[1]
        for i in range(dims):
            tmp_1=[]
            #print(inputSet[:,1])
            for article in inputSet[:,i]:
                tmp = [0] * len(vocabList)
                #print(article,"article")
               # for word in article:
                if article in vocabList:
                    tmp[vocabList.index(article)] = 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
                else:
                    print("the word :%s is not in my vocabulary" % article)
                tmp_1.append(tmp)
            returnVec.append(tmp_1)
        print(returnVec,"returnVec")
        return returnVec



    def fit(self, X, y):
            self.classes = np.unique(y)
            # 计算先验概率
            if self.class_prior == None:  # 不指定先验概率
                class_num = len(self.classes)  # 类别个数
                if not self.fit_prior:  # 是否自学先验概率，false则统一指定
                    self.class_prior = [1.0 / class_num for _ in range(class_num)]  # 统一优先级
                else:
                    self.class_prior = []
                    sample_num = float(len(y))
                    for c in self.classes:
                        c_num = np.sum(np.equal(y, c))
                        self.class_prior.append(
                            (c_num + self.alpha) / (sample_num + class_num * self.alpha))  # 根据标签出现个数计算优先级
                        # print(self.class_prior,"class_prior")
            # 计算条件概率
            self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{}, c1:{...} }
            numTrainDoc = len(X[0])  # 总共文档数量
            numWords = len(X[0][0])  # 单词数量
            pAbusive = sum(np.equal(y,-1)) / numTrainDoc  # 统计侮辱性文档总个数，然后除以总文档个数
            p0Num = np.ones(numWords)
            p1Num = np.ones(numWords)
            p0Denom = 2.0
            p1Denom = 2.0
            for c in self.classes:
                self.conditional_prob[c] = {}
                for i in range(len(X)):  # 特征总数2
                    for j in range(numTrainDoc):
                        if y[j] == -1:
                            p1Num += X[i][j]  # 把属于同一类的文本向量相加，实质是统计某个词条在该文本类中出现的频率
                            p1Denom += sum(X[i][j])  # 去重
                        else:
                            p0Num += X[i][j]
                            p0Denom += sum(X[i][j])
                    p1Vec = np.log(p1Num / p1Denom)  # 统计词典中所有词条在侮辱性文档中出现的概率
                    p0Vec = np.log(p0Num / p0Denom)  # 统计词典中所有词条在正常性文档中出现的概率
                    print(p1Vec,p0Vec,i)
            return pAbusive#, p1Vec, p0Vec

        # 给了 单词概率{value0:0.2,value1:0.1,value3:0.3,.. } 和目标值 给出目标值的概率
    def _get_xj_prob(self, values_prob, target_value):
            return values_prob.get(target_value)

        # 依据(class_prior,conditional_prob)预测一个简单地样本
    def _predict_single_sample(self, x):
            label = -1
            max_posterior_prob = 0
            # 对每一个类别，计算其后验概率：class_prior*conditional_prob
            for c_index in range(len(self.classes)):
                current_class_prior = self.class_prior[c_index]  # 类别优先级，类别多的优先级大
                current_conditional_prob = 1.0  # 条件概率
                feature_prob = self.conditional_prob[self.classes[c_index]]  # 类别1的条件概率
                # print(feature_prob.keys(),"feature_prob")
                j = 0
                for feature_i in feature_prob.keys():  # 每一个特征下的概率
                    current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i], x[j])
                    j += 1

                # 比较后验概率，更新max_posterior_prob ,label
                print(current_class_prior * current_conditional_prob, self.classes[c_index], "后验概率")
                if current_class_prior * current_conditional_prob > max_posterior_prob:  # 取最大的后验概率
                    max_posterior_prob = current_class_prior * current_conditional_prob
                    label = self.classes[c_index]

            return label

    def predict(self, X):
            if X.ndim == 1:
                return self._predict_single_sample(X)
            else:
                labels = []
                for i in range(X.shape(0)):
                    label = self._predict_single_sample(X[i])
                    labels.append(label)
                return labels

if __name__ == '__main__':
        X = np.array([
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
        ])
        X = X.T
        y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

        nb = BernoulliNB(alpha=1.0, fit_prior=True)
        vocabList =nb.createVocabList(X)
        returnVec=nb.setOfWord2Vec(vocabList,X)

        pAbusive=nb.fit(returnVec, y)
        print("[2,5]-->", nb.predict(np.array([2, 5])))  # 输出1 0.08169 0.04575
        print("[2,4]-->", nb.predict(np.array([2, 4])))  # 输出-1 0.0327 0.0610