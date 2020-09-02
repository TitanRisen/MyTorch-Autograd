# v0.1.0 说明 #
## 1. 基本说明  ##
本项目是张量的正向计算和对应的自动求导计算图的实现，接口参考了pytorch的Tensor和autograd的设计，测试通过了一维向量，二维矩阵以及三维张量的正向运算和对应的自动求导，简单使用可以查阅demo.ipynb。

## 2. 模块说明 ##
|文件名|说明|
|:-|:-:|
|demo.ipynb|简单的展示|
|Tensor.py|对应pytorch的Tensor及Variable，包含具体的如求导，log, sigmoid等操作|
|computeFlowSystem.py|计算图相关|
|matrixSystem.py|对张量（矩阵）的具体操作，目前主要实现了求迹的操作|
|Toolkit.py|工具包，包括对矩阵的遍历和随机采样等|
|grad/gradController.py|导数及导函数具体的定义及局部微分操作|
|grad/gradSystem.py|求导过程的具体实现|

## 3.设计说明 ##
* Tensor类继承自numpy的ndarray，接口设计参考了numpy和pytorch。
* 计算图的设计没有采用pytorch的动态计算图，而是参考了spark的RDD和DAG的设计，参考了其中的依赖关系，lineage等设计，后续可以加入并行算法。
* 求导计算有两种模式，如果函数的导函数存在于导函数字典里的话就直接调用，否则参考torch的求导方式，在邻域采样算斜率。
* 本意上是pytorch的tensor和计算图的学习和轻量版实现，由于加入了导函数字典，所以在导函数已知时进行BP在单CPU线程的情况下算cpu时间比torch稍快，但是做不到cuda的并行化...
* 这个版本基本不更新了，有兴趣的同学可以继续加入conv,pooling和bptt等算法，在上层创建网络结构封装成网络层就能当轻量版的pytorch来用

## 4.参考资料 ##
* https://zhuanlan.zhihu.com/p/24709748
* http://spark.apachecn.org/
* https://www.cnblogs.com/catnip/p/8760780.html
* https://pytorch-cn.readthedocs.io/zh/latest/
* Stanford University CS231n by Feifei Li
* ...