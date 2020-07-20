# datawhale_nlp
#比赛链接:https://tianchi.aliyun.com/competition/entrance/531810/introduction

第一次接触NLP方面的比赛，之前接触的比赛大多是结构化赛题，本着提升自己的目标，加入了组队学习，希望能够以赛促学，对nlp问题，以及深度学习方面有所了解。
话不多说 直接开始学习笔记。

首先是数据集的获取
在官网上的数据集下载后是对应数据的链接。

训练集看似不大（800M） 实则800M不过是冰山一角，为了防止人工打标的情况发生，赛方已经分词完毕，并将分词结果匿名化，只能通过匿名后的标签进行训练与预测。
首先，使用pandas.str.split(' ')将dataframe中的text转化为列表形式，起初想直接通过expand=True直接展开，当场内存爆炸（MemoryError）

一般非结构化的如nlp、cv方面，采用的方法均为对其进行特征提取，使其转化为结构化数据，机器学习中常见的转化有CountVectorizer、TfidfVectorizer（词袋模型、逆文档频率模型）两种词向量转化方法。
根据之前的些许经验，我直接选择了词袋模型（事后再次线下验证，确实词袋模型效果更好）
