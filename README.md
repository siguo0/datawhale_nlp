# datawhale_nlp
#比赛链接:https://tianchi.aliyun.com/competition/entrance/531810/introduction

第一次接触NLP方面的比赛，之前接触的比赛大多是结构化赛题，本着提升自己的目标，加入了组队学习，希望能够以赛促学，对nlp问题，以及深度学习方面有所了解。
话不多说 直接开始学习笔记。

首先是数据集的获取
在官网上的数据集下载后是对应数据的链接。

首先观察题目，是一个多分类问题，训练集看似不大（800M） 实则800M不过是冰山一角，为了防止人工打标的情况发生，赛方已经分词完毕，并将分词结果匿名化，只能通过匿名后的标签进行训练与预测。
首先，使用pandas.str.split(' ')将dataframe中的text转化为列表形式，起初想直接通过expand=True直接展开，当场内存爆炸（MemoryError）

一般非结构化的如nlp、cv方面，采用的方法均为对其进行特征提取，使其转化为结构化数据，机器学习中常见的转化有CountVectorizer、TfidfVectorizer（词袋模型、逆文档频率模型）两种词向量转化方法。
根据之前的些许经验，我直接选择了词袋模型（事后再次线下验证，确实词袋模型效果更好）
词袋模型：统计词频，将词作为特征，每个文档的该词词频为特征值
TFIDF模型：词频-逆文档频率，通过TF-词频，DF-文档频数通过一定方法计算，得到的值为该词在该文档中的权重

为了内存着想，设置词袋模型中的参数，取一定范围，min_df,max_df,设置后训练转化训练集（此处我直接对于训练集进行训练，不知道是否将测试集加入训练效果会更好？尚未尝试）

转化后的特征维度有1000+，通过train_test_split按照8:2划分训练测试集

线下首先尝试了一下多项式朴素贝叶斯模型，线下验证效果0.77左右，放弃

其后尝试lightgbm模型，线下验证0.91左右，提交，与线上基本一致，到此0.91 baseline完成
对lightgbm进行简单调参--仅调整迭代次数，用验证集做eval_set,在3、400次迭代时损失达到最低，线下验证0.93左右 线上提交0.93
