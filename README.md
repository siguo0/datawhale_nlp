# datawhale_nlp
#比赛链接:https://tianchi.aliyun.com/competition/entrance/531810/introduction

第一次接触NLP方面的比赛，之前接触的比赛大多是结构化赛题，本着提升自己的目标，加入了组队学习，希望能够以赛促学，对nlp问题，以及深度学习方面有所了解。
话不多说 直接开始学习笔记。

**首先是数据集的获取**
>在官网上的数据集下载后是对应数据的链接。

**观察题目，是一个多分类问题**
>训练集看似不大（800M） 实则800M不过是冰山一角，为了防止人工打标的情况发生，赛方已经分词完毕，并将分词结果匿名化，只能通过匿名后的标签进行训练与预测。
>使用pandas.str.split(' ')将dataframe中的text转化为列表形式，起初想直接通过expand=True直接展开，当场内存爆炸（MemoryError）

**特征提取**
>1.一般非结构化的如nlp、cv方面，采用的方法均为对其进行特征提取，使其转化为结构化数据<br>
>2.机器学习中常见的转化有CountVectorizer、TfidfVectorizer（词袋模型、逆文档频率模型）两种词向量转化方法。
>>词袋模型：统计词频，将词作为特征，每个文档的该词词频为特征值<br>
>>TFIDF模型：词频-逆文档频率，通过TF-词频，DF-文档频数通过一定方法计算，得到的值为该词在该文档中的权重

>根据之前的些许经验，我直接选择了词袋模型（事后再次线下验证，确实词袋模型效果更好）

**baseline的编写**
>***特征构造***
<br>为了内存着想，设置词袋模型中的参数，取一定范围，min_df,max_df,设置后训练转化训练集（此处我直接对于训练集进行训练，不知道是否将测试集加入训练效果会更好？尚未尝试）
<br>转化后的特征维度有1000+，通过train_test_split按照8:2划分训练测试集

>***模型选择***
<br>1.线下首先尝试了一下多项式朴素贝叶斯模型，线下验证效果0.77左右，放弃
<br>2.其后尝试lightgbm模型，线下验证0.91左右，提交，与线上基本一致，到此0.91 baseline完成
<br>3.对lightgbm进行简单调参--仅调整迭代次数，用验证集做eval_set,在3、400次迭代时损失达到最低，线下验证0.93左右 线上提交0.93


TASK1
<br>赛题理解：
----
<br>

**解题思路**

>1、赛题本质是一个文本分类问题，需要根据每句的字符对文章进行分类。但赛题给出的数据是匿名化的，相当于赛方已经完成分词工作,
由于文本数据是一种典型的非结构化数据，要通过模型进行分类的话，需要通过特征提取工作获得可训练的数据集。

>2、思路：词袋模型+lightgbm
>>词袋模型:根据统计文章词频，将词频作为对应特征的一种文本结构化方式
<br>lightgbm:这是机器学习中速度与准确率综合相对较好的一种模型

TASK2
<br>数据分析：
---
<br>
1、观察训练集标签分布

```
train.label.plot(kind='hist')
```

>可观察到0类别标签数量最多

2、其次对text字段使用split方法，将其划分为列表形式
```
train.text=train.text.str.split(' ')
```

3、观察每篇文章的长度

```
train['len']=list(map(lambda x:len(x),train.text))
train.len.describe()
```

4、将文章长度进行筛选绘制

```
train[train.len<10000].len.plot(kind='hist')
```

>可以看到文章长度大多在0-2000以内


TASK3
<br>：基于机器学习的文本分类
---
<br>
1、首先导入词袋模型和TFIDF模型,选择词袋模型，设置参数，以下参数仅供64G内存运行，32G内存极限max_features应为10000左右，内存不够用时，对数据类型进行变化，能够有效降低内存，词袋模型中记录为词频，一般而言，词频不会太大,将其转化为uint8类型。

```
def word2list(list_word):
    word_list=[]
    for i in list_word:
        word_list.append(','.join(i))
    return word_list
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
ngram_vectorizer = CountVectorizer(ngram_range=(1,4),min_df=15,max_df=0.8,max_features=20000,decode_error='replace')
word_list=word2list(train.text.str.split(' '))
count=ngram_vectorizer.fit(word_list)

word_list=count.transform(word_list)
label=train.label
train=pd.DataFrame(word_list.toarray(),dtype=np.uint8)
train['label']=label
del word_list
```

>ngram_range参数作用相当于滑窗，可在一定程度上考虑到文章上下文相关性，但也会大大加重内存负担。

>min_df设定最少在多少篇文章中出现，才保留该单词，max_df同理。

>max_features设定最终保留单词的数量，即维度，设置后，它会根据词频从大到小的原则进行截取。


2、按照8:2的比例划分训练测试集

```
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(train.drop('label',axis=1),train.label,test_size=0.2,random_state=2020)
del train
```

3、导入lightgbm模型进行训练预测

```
from lightgbm import LGBMClassifier

gbm=LGBMClassifier(n_estimators=100000)
gbm.fit(train_x,train_y,eval_set=[(test_x,test_y)],verbose=True,early_stopping_rounds=15)
pre=gbm.predict(test_x)
f1_score(test_y,pre, average='macro')
```

>eval_set设置测试集为标准,当测试集的损失到达一定程度，无法在下降时，early_stopping_rounds会使模型迭代提前结束

4、线上分数

>随着max_features的逐渐增大，模型的准确率会有所提高，直至该方法的上限，以上代码的线上成绩为0.947,模型上限应该在0.95左右，但是太过耗费资源，16G内存应该上限在0.94左右，32G可以勉强达到0.945。


TASK4
<br>：基于深度学习的文本分类
---
<br>
1、基于深度学习进行文本分类，本次采用的模型为fasttext模型
