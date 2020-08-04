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

>fasttext是一种快速文本分类的算法,在fasttext中能够使用分层softmax方法加快运行速度，普通softmax需要对所有类别的概率进行归一化,在类别较大的情况下这是比较耗时的，分层softmax根据类别的频率构建哈夫曼树，能够减少计算量。

>n-gram能够设定滑窗，在一定程度上可以认为考虑上下文信息或者单词共现特征。

>内存问题：fasttext使用哈希桶的方式，将所有n-gram共享一个embedding vector,既保证了效率又减少内存的占用

>fasttext虽然可能在准确率方面对于词袋模型略有不如,线下验证在0.92左右,但这是仅通过简单调参,得到的结果,词袋模型中有参数可以直接设置词频的获取,如min_df,max_df本身就相当于一层筛选，而对于fasttext模型我暂时未进行数据清洗,并且fasttext的优点在于其能够快速进行,相比词袋模型,不仅大大减少内存的使用,并且节约了许多的时间

---

```
import pandas as pd
from sklearn.metrics import f1_score
import fasttext
# 转换为FastText需要的格式
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=200000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-20000].to_csv('train.csv', index=None, header=None, sep='\t')

```

>导入第三方库,对原始数据进行格式改变,fasttext需要标签为__label__的形式

```
model = fasttext.train_supervised('train.csv', lr=1, wordNgrams=3,
                                  verbose=2, minCount=1, epoch=100, loss="hs",dim=500)
```

>初始化fasttext算法模型,传入的数据应该为一个文件,fasttext会通过wordembedding的方法将文章映射到向量空间中,此处损失使用'hs'分层softmax方法，可以大大加快运行速度

```
val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-20000:]['text']]
print(f1_score(train_df['label'].values[-20000:].astype(str), val_pred, average='macro'))
```

>简单地对模型结果进行验证,线下成绩为0.92左右



TASK5
<br>基于深度学习的文本分类:
---

**一、word2vec方法**

>word2vec是一种词嵌入方法,本身还是一种单词向量化的方法,根据已有的单词,进行训练,通过神经网络的权重来对单词进行表示，使非结构化的单词变化为结构化的数据形式,本质上相当于一种对于one-hot编码的降维算法

>本次word2vec模型,使用gensim进行训练

```
from gensim.models.word2vec import Word2Vec

num_features = 100     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel

train_texts = list(map(lambda x: list(x.split()), train_texts))
model = Word2Vec(train_texts, workers=num_workers, size=num_features)
model.init_sims(replace=True)

# save model
model.save("./word2vec.bin")
```

>整体搬运学习资料中的方法,学习资料中,设定维度为100维,只选取了部分数据进行训练,10000条样本数据,单词数为4500+。

>word2vec基本思想是对出现在上下文环境里的词进行预测,因此它本身就考虑了上下文关系,相对于N-gram算法,word2vec的考虑窗口更大。

>由于word2vec要计算所有单词的权重,随着单词的增加,计算量逐渐增大,因此需要训练技巧来加速训练过程
>>negative sampling:本质是预测总体类别的一个子集
>>hierarchical softmax:本质是把 N 分类问题变成 log(N)次二分类

>word2vec相当于一种词嵌入方法,其本身并无法直接对文本问题进行预测,主要应用为预训练模型,将非结构化的单词数据转化为结构化的数值。

**二:textCNN方法**

>对于深度学习方法相当于零基础,因此只能对原理进行一点分析

>卷积神经网络的核心思想是捕捉局部特征，对于文本来说，局部特征就是由若干单词组成的滑动窗口，类似于N-gram。卷积神经网络的优势在于能够自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。(摘录)

>textcnn中首先进行一个embedding词嵌入,将文字嵌入到向量中作为模型卷积层的输入,示例中使用的即为预训练好的word2vec模型

**三、textRNN方法**

>TextRNN是将Word Embedding输入到双向LSTM中，然后对最后一位的输出输入到全连接层中，在对其进行softmax分类即可。

>关于RNN与LSTM暂时并无太多了解,后续补充


TASK6
<br>基于深度学习的文本分类3-BERT:
---
**bert概念**

>bert是一种基于预训练的模型,其基于语言模型,在其基础上引入马尔科夫假设的n-gram模型

>BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder

>bert的主要创新方法在于预训练方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

>bert使用transformers模型替代传统RNN模型作为seq2seq的基础，其中transformers模型使用self-attention注意力机制，transformer模型相对于传统RNN模型更加高效，能够捕捉更长距离依赖。

**bert对于该题的效果**

>据说20w数据能够基本完成bert模型的预训练,不过bert模型需要耗费较多时间，当前个人笔记本另做它用，无法同时进行，因此未能完整尝试，不过baseline中10000数据的效果似乎不佳？是否因为bert模型在缺乏训练时自然效果不佳，或者自己操作失误？