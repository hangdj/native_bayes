# coding:utf-8
import numpy
def loadDataSet():  # 构建一个简单的文本集，以及标签信息 1 表示侮辱性文档，0表示正常文档
        postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec

def createVocabList(dataSet):  # 统计词汇，创建词典
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        #print (vocabSet)
        return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):  # 创建词频统计向量,此方法为词集模型，出现赋值1，or0
        returnVec = [] # 返回的文本向量与词典大小保持一致
        for article in inputSet:
            tmp = [0] * len(vocabList)
            for word in article:
                if word in vocabList:
                    tmp[vocabList.index(word)] = 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
                else:
                    print ("the word :%s is not in my vocabulary" % word)
            returnVec.append(tmp)
        print(returnVec)
        return returnVec

def bagOfWord2Vec(vocabList, inputSet):  # 词袋模型，统计概率的
        returnVec = []
        for article in inputSet:
            tmp=[0]*len(vocabList)
            for word in article:
                if word in vocabList:
                    tmp[vocabList.index(word)] += 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
                else:
                    print ("the word :%s is not in my vocabulary" % word)
            returnVec.append(tmp)
        print (returnVec)
        return returnVec
def trianNB(trainMatrix,trainCategory):#训练生成朴素贝叶斯模型
    '''

    :param trainMatrix: 训练数组
    :param trainCategory: 训练标签
    :return:
    '''
    numTrainDoc=len(trainMatrix)#总共文档数量
    numWords=len(trainMatrix[0])#单词数量
    pAbusive=sum(trainCategory)/numTrainDoc#统计侮辱性文档总个数，然后除以总文档个数
    p0Num=numpy.ones(numWords)
    p1Num=numpy.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDoc):
        if trainCategory[i]==1:#如果是侮辱性文档
            p1Num+=trainMatrix[i]#把属于同一类的文本向量相加，实质是统计某个词条在该文本类中出现的频率
            p1Denom+=sum(trainMatrix[i])#去重
            print(p1Denom,"p1Denom")
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
            print(p0Denom,"p0Denom")
    p1Vec=numpy.log(p1Num/p1Denom)#统计词典中所有词条在侮辱性文档中出现的概率
    p0Vec=numpy.log(p0Num/p0Denom)#统计词典中所有词条在正常性文档中出现的概率
    return pAbusive,p1Vec,p0Vec
def classifyNB(vec2classify,p0Vec,p1Vec,pClass1):# 参数1是测试文档向量，参数2和参数3是词条在各个
                                                    #类别中出现的概率，参数4是P（C1）
    p1=numpy.sum(vec2classify*p1Vec)+numpy.log(pClass1)
    p0=numpy.sum(vec2classify*p0Vec)+numpy.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
if __name__=='__main__':
    test=[['mr', 'licks', 'ate', 'my', 'steak', 'how', 'food', 's']]
    #test=[['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']]
    postingList, classVec=loadDataSet()#文档，标签
    vocabList=createVocabList(postingList)#词典
    returnVec=bagOfWord2Vec(vocabList,postingList)#文本向量
    pAbusive, p1Vec, p0Vec=trianNB(returnVec,classVec)#侮辱性文档比例，侮辱文档概率，正常文档概率
    print (pAbusive, p1Vec, p0Vec)
    testVec=bagOfWord2Vec(vocabList,test)
    pclass=classifyNB(testVec,p0Vec,p1Vec,pAbusive)
    print (pclass)
'''
{1: {0: {1: 0.25, 2: 0.3333333333333333, 3: 0.4166666666666667}, 1: {4: 0.16666666666666666, 5: 0.4166666666666667, 6: 0.4166666666666667}},
 -1: {0: {1: 0.4444444444444444, 2: 0.3333333333333333, 3: 0.2222222222222222}, 1: {4: 0.4444444444444444, 5: 0.3333333333333333, 6: 0.2222222222222222}}} conditional_prob
'''