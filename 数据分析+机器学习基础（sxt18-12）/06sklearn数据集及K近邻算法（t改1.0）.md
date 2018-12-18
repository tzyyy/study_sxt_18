## 机器学习应用程序的步骤

**（1）收集数据**

我们可以使用很多方法收集样本护具，如：

公司自有数据

制作网络爬虫从网站上抽取数据、

第三方购买的数据

合作机构提供的数据

从RSS反馈或者API中得到信息、设备发送过来的实测数据。

**（2）准备输入数据**

得到数据之后，还必须确保数据格式符合要求。

**（3）分析输入数据**

这一步的主要作用是确保数据集中没有垃圾数据。如果是使用信任的数据来源，那么可以直接跳过这个步骤

**（4）训练算法**

机器学习算法从这一步才真正开始学习。如果使用无监督学习算法，由于不存在目标变量值，故而也不需要训练算法，所有与算法相关的内容在第（5）步

**（5）测试算法**

这一步将实际使用第（4）步机器学习得到的知识信息。当然在这也需要评估结果的准确率，然后根据需要重新训练你的算法

**（6）使用算法**

转化为应用程序，执行实际任务。以检验上述步骤是否可以在实际环境中正常工作。如果碰到新的数据问题，同样需要重复执行上述的步骤

# 一、sklearn数据集

### 1、sklearn中的练习数据集

**1.1、datasets.load_\()**

获取小规模数据集，数据包含在datasets里

```python
#小数据集 
#波士顿房价数据集        load_boston       回归            数据量：50613 
#鸢尾花数据集           load_iris         分类            数据量：1504 
#糖尿病数据集           load_diabetes     回归            数据量： 4210 
#手写数字数据集         load_digits       分类            数据量：562064 
..........
datasets.load_boston（[return_X_y]）	加载并返回波士顿房价数据集（回归）。
datasets.load_breast_cancer（[return_X_y]）加载并返回乳腺癌威斯康星数据集（分类）。
datasets.load_diabetes（[return_X_y]）加载并返回糖尿病数据集（回归）。
datasets.load_digits（[n_class，return_X_y]）	加载并返回数字数据集（分类）。
datasets.load_files（container_path [，...]）加载带有类别的文本文件作为子文件夹名称。
datasets.load_iris（[return_X_y]）	加载并返回虹膜数据集（分类）。
datasets.load_linnerud（[return_X_y]）	加载并返回linnerud数据集（多元回归）。
datasets.load_sample_image（IMAGE_NAME）	加载单个样本图像的numpy数组
datasets.load_sample_images（）	加载样本图像以进行图像处理。
datasets.load_svmlight_file（f [，n_features，...]）将svmlight / libsvm格式的数据集加载到稀疏CSR矩阵中
datasets.load_svmlight_files（文件[，...]）	以SVMlight格式从多个文件加载数据集
datasets.load_wine（[return_X_y]）	加载并返回葡萄酒数据集（分类）。
```

**1.2datasets.fetch_\()**

获取大规模数据集，需要从网络上下载，函数的第一个参数是data_home，表示数据集下载的目录，默认是 ~/scikit_learn_data/，要修改默认目录，可以修改环境变量SCIKIT_LEARN_DATA

**数据集目录可以通过datasets.get_data_home()获取，clear_data_home(data_home=None)删除所有下载数据**

```python
#大数据集 
#Olivetti 脸部图像数据集       fetch_olivetti_faces      降维            
#新闻分类数据集                fetch_20newsgroups        分类
#带标签的人脸数据集             fetch_lfw_people          分类；降维 -
#加州房价数据					fetch_california_housing    回归
........
```

**1.3datasets.make_\()**

### 2、sklearn中数据集的属性

**load\*和 fetch* 函数返回的数据类型是 datasets.base.Bunch，本质上是一个 dict，它的键值对可用通过对象的属性方式访问。主要包含以下属性：**

- data：特征数据数组，是 n_samples ，n_features 的二维 numpy.ndarray 数组
- target：标签数组，是 n_samples 的一维 numpy.ndarray 数组
- DESCR：数据描述
- feature_names：特征名
- target_names：标签名

### 3、获取小数据集

```python
def get_data():
    #数据集获取(糖尿病数据集)  #回归数据
    li = load_diabetes()
    #获取特征数据
    print(li.data)
    #获取目标值
    print(li.target)
    #获取描述信息
    print(li.DESCR)
    
    
 def get_data():
    #数据集获取(糖尿病数据集)  #分类数据
    li = load_iris()
    #获取特征数据
    print(li.data)
    #获取目标值
    print(li.target)
    #获取描述信息
    print(li.DESCR)
    

```

### 4、获取大数据集

```python
def get_news():
    # 获取分类数据（新闻）
    news = fetch_20newsgroups(subset='all')
    print(news.DESCR)
    print(len(news.data))
    
#可选参数：

#subset: 'train'或者'test','all'，可选，选择要加载的数据集：训练集的“训练”，测试集的“测试”，两者的“全部”，具有洗牌顺序

#data_home: 可选，默认值：无，指定数据集的下载和缓存文件夹。如果没有，所有scikit学习数据都存储在'〜/ scikit_learn_data'子文件夹中

#categories: 无或字符串或Unicode的集合，如果没有（默认），加载所有类别。如果不是无，要加载的类别名称列表（忽略其他类别）

#shuffle: 是否对数据进行洗牌

#random_state: numpy随机数生成器或种子整数

#download_if_missing: 可选，默认为True，如果False，如果数据不在本地可用而不是尝试从源站点下载数据，则引发IOError

```

### 5、获取本地生成数据

##### 1、生成本地分类数据：

- sklearn.datasets.make_classification
- make_multilabel_classification

```python
主要参数
"""
生成用于分类的数据集

:param n_samples:int，optional（default = 100)，样本数量

:param n_features:int，可选（默认= 20），特征总数

:param n_classes:int，可选（default = 2),类（或标签）的分类问题的数量

:param random_state:int，RandomState实例或无，可选（默认=无）

返回值
:return :X,特征数据集；y,目标分类值
"""
```

案列

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_classification
# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()
```

##### 2、生成本地回归模型数据：

- sklearn.datasets.make_regression

```python
  """
  生成用于回归的数据集

  :param n_samples:int，optional（default = 100)，样本数量

  :param  n_features:int,optional（default = 100)，特征数量

  :param  coef:boolean，optional（default = False），如果为True，则返回底层线性模型的系数

  :param random_state:随机数生成器使用的种子;
  
  :return :X,特征数据集；y,目标值
  """
```

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_regression
# X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
X, y, coef =make_regression(n_samples=1000, n_features=1,noise=10, coef=True)
# 画图
plt.scatter(X, y,  color='black')
plt.plot(X, X*coef, color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```



##### 3、生成本地聚类模型数据

- sklearn.datasets.make_blobs

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()
```



##### 4、 生成本地分组多维正态分布的数据

- sklearn.datasets.make_gaussian_quantiles

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_gaussian_quantiles
#生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
```



### 6、数据分割

```python
from sklearn.datasets import load_boston

def split_data():
    '''数据集分割'''
    #获取(波士顿数据集)
   	li = load_bosto()
    #对数据进行分割
    result = train_test_split(li.data,li.target,test_size=0.25)
    # result接收的是一个元祖（训练集，目标值，测试集，目标值）
    x_tran,x_target,y_tran,y_target = result
    print('训练集数据',x_tran)
    print('训练集目标值', x_tran)
```

# 二、有监督学习和无监督学习

算法是核心，数据和计算是基础。

- 监督学习
  - 分类 k-近邻算法、决策树、贝叶斯、逻辑回归(LR)、支持向量机(SVM)
  - 回归 线性回归、岭回归
- 无监督学习
  - 聚类 k-means

### 1、监督学习（supervised learning）

从给定的训练数据集中学习出一个函数（模型参数），当新的数据到来时，可以根据这个函数预测结果。监督学习的训练集要求包括输入输出，也可以说是特征和目标。训练集中的目标是由人标注的。监督学习就是最常见的分类（注意和聚类区分）问题，通过已有的训练样本（即已知数据及其对应的输出）去训练得到一个最优模型（这个模型属于某个函数的集合，最优表示某个评价准则下是最佳的），再利用这个模型将所有的输入映射为相应的输出，对输出进行简单的判断从而实现分类的目的。也就具有了对未知数据分类的能力。监督学习的目标往往是让计算机去学习我们已经创建好的分类系统（模型）。

监督学习是训练神经网络和决策树的常见技术。这两种技术高度依赖事先确定的分类系统给出的信息，对于神经网络，分类系统利用信息判断网络的错误，然后不断调整网络参数。对于决策树，分类系统用它来判断哪些属性提供了最多的信息。

**常见的有监督学习算法：回归分析和统计分类。最典型的算法是KNN和SVM**



### 2、无监督学习（unsupervised learning）

输入数据没有被标记，也没有确定的结果。样本数据类别未知，需要根据样本间的相似性对样本集进行分类（聚类，clustering）试图使类内差距最小化，类间差距最大化。通俗点将就是实际应用中，不少情况下无法预先知道样本的标签，也就是说没有训练样本对应的类别，因而只能从原先没有样本标签的样本集开始学习分类器设计。

非监督学习目标不是告诉计算机怎么做，而是让它（计算机）自己去学习怎样做事情。非监督学习有两种思路。第一种思路是在指导Agent时不为其指定明确分类，而是在成功时，采用某种形式的激励制度。需要注意的是，这类训练通常会置于决策问题的框架里，因为它的目标不是为了产生一个分类系统，而是做出最大回报的决定，这种思路很好的概括了现实世界，agent可以对正确的行为做出激励，而对错误行为做出惩罚。

无监督学习的方法分为两大类：

(1)    一类为基于概率密度函数估计的直接方法：指设法找到各类别在特征空间的分布参数，再进行分类。

(2)    另一类是称为基于样本间相似性度量的简洁聚类方法：其原理是设法定出不同类别的核心或初始内核，然后依据样本与核心之间的相似性度量将样本聚集成不同的类别。

利用聚类结果，可以提取数据集中隐藏信息，对未来数据进行分类和预测。应用于数据挖掘，模式识别，图像处理等。

    PCA和很多deep learning算法都属于无监督学习。 
### 两者的不同点

1.      有监督学习方法必须要有训练集与测试样本。在训练集中找规律，而对测试样本使用这种规律。而非监督学习没有训练集，只有一组数据，在该组数据集内寻找规律。

2.      有监督学习的方法就是识别事物，识别的结果表现在给待识别数据加上了标签。因此训练样本集必须由带标签的样本组成。而非监督学习方法只有要分析的数据集的本身，预先没有什么标签。如果发现数据集呈现某种聚集性，则可按自然的聚集性分类，但不予以某种预先分类标签对上号为目的。

3.      非监督学习方法在寻找数据集中的规律性，这种规律性并不一定要达到划分数据集的目的，也就是说不一定要“分类”。

这一点是比有监督学习方法的用途要广。    譬如分析一堆数据的主分量，或分析数据集有什么特点都可以归于非监督学习方法的范畴。

4.      用非监督学习方法分析数据集的主分量与用K-L变换计算数据集的主分量又有区别。后者从方法上讲不是学习方法。因此用K-L变换找主分量不属于无监督学习方法，即方法上不是。而通过学习逐渐找到规律性这体现了学习方法这一点。在人工神经元网络中寻找主分量的方法属于无监督学习方法。
# 三、sklearn分类算法之k-近邻

k-近邻算法采用测量不同特征值之间的距离来进行分类

> 优点：精度高、对异常值不敏感、无数据输入假定
>
> 缺点：计算复杂度高、空间复杂度高
>
> 使用数据范围：数值型和标称型

### 1、k-近邻法简介

k近邻法(k-nearest neighbor, k-NN)是1967年由Cover T和Hart P提出的一种基本分类与回归方法。它的工作原理是：存在一个样本数据集合，也称作为训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一个数据与所属分类的对应关系。输入没有标签的新数据后，将新的数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本最相似数据(最近邻)的分类标签。一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

举个简单的例子，我们可以使用k-近邻算法分类一个电影是爱情片还是动作片。

![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](https://cuijiahua.com/wp-content/uploads/2017/11/ml_1_1.png)

表1.1 每部电影的打斗镜头数、接吻镜头数以及电影类型

表1.1 就是我们已有的数据集合，也就是训练样本集。这个数据集有两个特征，即打斗镜头数和接吻镜头数。除此之外，我们也知道每个电影的所属类型，即分类标签。用肉眼粗略地观察，接吻镜头多的，是爱情片。打斗镜头多的，是动作片。以我们多年的看片经验，这个分类还算合理。如果现在给我一部电影，你告诉我这个电影打斗镜头数和接吻镜头数。不告诉我这个电影类型，我可以根据你给我的信息进行判断，这个电影是属于爱情片还是动作片。而k-近邻算法也可以像我们人一样做到这一点，不同的地方在于，我们的经验更"牛逼"，而k-近邻算法是靠已有的数据。比如，你告诉我这个电影打斗镜头数为2，接吻镜头数为102，我的经验会告诉你这个是爱情片，k-近邻算法也会告诉你这个是爱情片。你又告诉我另一个电影打斗镜头数为49，接吻镜头数为51，我"邪恶"的经验可能会告诉你，这有可能是个"爱情动作片"，画面太美，我不敢想象。 (如果说，你不知道"爱情动作片"是什么？请评论留言与我联系，我需要你这样像我一样纯洁的朋友。) 但是k-近邻算法不会告诉你这些，因为在它的眼里，电影类型只有爱情片和动作片，它会提取样本集中特征最相似数据(最邻近)的分类标签，得到的结果可能是爱情片，也可能是动作片，但绝不会是"爱情动作片"。当然，这些取决于数据集的大小以及最近邻的判断标准等因素。

### 2、sklearn.neighbors.

**sklearn.neighbors.KNeighborsClassifier**

sklearn.neighbors提供监督的基于邻居的学习方法的功能，

sklearn.neighbors.KNeighborsClassifier是一个最近邻居分类器。那么KNeighborsClassifier是一个类，我们看一下实例化时候的参数

##### 1、参数

```python
 """
  :param n_neighbors：int，可选（默认= 5），k_neighbors查询默认使用的邻居数
  :param algorithm：{'auto'，'ball_tree'，'kd_tree'，'brute'}，可选用于计算最近邻居的算法：
  'ball_tree'将会使用 BallTree，
  'kd_tree'将使用 KDTree，“野兽”将使用强力搜索。
  'auto'将尝试根据传递给fit方法的值来决定最合适的算法。
  :param n_jobs：int，可选（默认= 1),用于邻居搜索的并行作业数。如果-1
"""
```

##### 2、方法

**fit(X, y)**

使用X作为训练数据拟合模型，y作为X的类别值。X，y为数组或者矩阵

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
X = np.array([[1,1],[1,1.1],[0,0],[0,0.1]])
y = np.array([1,1,0,0])
neigh.fit(X,y)
```

**kneighbors(X=None, n_neighbors=None, return_distance=True)**

找到指定点集X的n_neighbors个邻居，return_distance为False的话，不返回距离

```python
neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False)

neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False,an_neighbors=2)
```

**predict(X)**

预测提供的数据的类标签

```python
neigh.predict(np.array([[0.1,0.1],[1.1,1.1]]))
```

**predict_proba(X)**

返回测试数据X属于某一类别的概率估计

```python
neigh.predict_proba(np.array([[1.1,1.1]]))
```

### 3、GridSearchCV

sklearn里面的GridSearchCV用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。

​	它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再拿bagging再优化。



```python
classsklearn.model_selection.GridSearchCV(estimator,
                                          param_grid, 
                                          scoring=None, 
                                          fit_params=None,
                                          n_jobs=1,
                                          iid=True,
                                          refit=True,
                                          cv=None, 
                                          verbose=0, 
                                          pre_dispatch='2*n_jobs',
                                          error_score='raise',
                                          return_train_score=True)
```



##### 1、常用参数

```python
'''
estimator：所使用的分类器，如estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10), 并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法。

param_grid：值为字典或者列表，即需要最优化的参数的取值，
param_grid =param_test1，
param_test1 = {'n_estimators':range(10,71,10)}。

scoring :准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。

cv :交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。

refit :默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。

iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。

verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。

pre_dispatch：指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次

'''
```

##### 2、进行预测的常用方法和属性

```python
'''
grid.fit()：运行网格搜索

grid_scores_：给出不同参数情况下的评价结果

best_params_：描述了已取得最佳结果的参数的组合

score(x_test, y_test)：在测试集上准确率

best_score_ ：在交叉验证当中最好的结果

gs.best_estimator_ ：选择最好的模型是：

gs.cv_results_  ：每个超参数每次交叉验证的结果
'''
```



##### 3、案列一

```python
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

def knniris():
    """
    鸢尾花分类
    :return: None
    """

    # 数据集获取和分割
    lr = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.25)

    # 进行标准化
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # estimator流程
    knn = KNeighborsClassifier()

    # 通过网格搜索,n_neighbors为参数列表
    param = {"n_neighbors": [3, 5, 7]}

    gs = GridSearchCV(knn, param_grid=param, cv=10)

    # 建立模型
    gs.fit(x_train,y_train)

    # 预测准确率
    print("在测试集上准确率：", gs.score(x_test, y_test))

    print("在交叉验证当中最好的结果：", gs.best_score_)

    print("选择最好的模型是：", gs.best_estimator_)

    print("每个超参数每次交叉验证的结果：", gs.cv_results_)


knniris()
```

结果：

```
在测试集上准确率： 1.0
在交叉验证当中最好的结果： 0.9375
选择最好的模型是： KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
每个超参数每次交叉验证的结果： {'mean_fit_time': array([0.00135021, 0.00095284, 0.00100255]), 'std_fit_time': array([0.00059321, 0.0002651 , 0.00022308]), 'mean_score_time': array([0.00220425, 0.001596  , 0.00164721]), 'std_score_time': array([0.00074352, 0.00037176, 0.0002317 ]), 'param_n_neighbors': masked_array(data=[3, 5, 7],
             mask=[False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}], 'split0_test_score': array([0.91666667, 0.83333333, 0.83333333]), 'split1_test_score': array([0.91666667, 1.        , 1.        ]), 'split2_test_score': array([1., 1., 1.]), 'split3_test_score': array([0.83333333, 0.91666667, 0.91666667]), 'split4_test_score': array([0.83333333, 0.91666667, 0.91666667]), 'split5_test_score': array([0.91666667, 1.        , 0.91666667]), 'split6_test_score': array([1.        , 1.        , 0.90909091]), 'split7_test_score': array([0.81818182, 0.81818182, 0.81818182]), 'split8_test_score': array([0.88888889, 0.88888889, 0.88888889]), 'split9_test_score': array([1., 1., 1.]), 'mean_test_score': array([0.91071429, 0.9375    , 0.91964286]), 'std_test_score': array([0.06671883, 0.06925931, 0.06165686]), 'rank_test_score': array([3, 1, 2]), 'split0_train_score': array([0.94, 0.95, 0.95]), 'split1_train_score': array([0.96, 0.95, 0.95]), 'split2_train_score': array([0.93, 0.95, 0.94]), 'split3_train_score': array([0.95, 0.94, 0.95]), 'split4_train_score': array([0.95, 0.96, 0.94]), 'split5_train_score': array([0.95, 0.97, 0.94]), 'split6_train_score': array([0.94059406, 0.96039604, 0.94059406]), 'split7_train_score': array([0.94059406, 0.95049505, 0.96039604]), 'split8_train_score': array([0.95145631, 0.97087379, 0.95145631]), 'split9_train_score': array([0.94174757, 0.95145631, 0.94174757]), 'mean_train_score': array([0.9454392 , 0.95532212, 0.9464194 ]), 'std_train_score': array([0.00799474, 0.00928929, 0.00662243])}

```



##### 4、案列二

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_classification

X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
plt.show()

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors = 15 , weights='distance')
clf.fit(X, Y)

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#确认训练集的边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#生成随机数据来做测试集，然后作预测
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 画出测试集数据
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
```

