windows和linux下xgboost安装指南:  http://blog.csdn.net/xizero00/article/details/73008330

基于之前建立的基学习器的损失函数的梯度下降方向来建立下一个新的基学习器，目的就是希望通过集成这些基学习器使得模型总体的损失函数不断下降，模型不断改进。

### 一、xgboost的优点？
- 高度的灵活性
传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。

-  正则化
xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性。

- Shrinkage（缩减）
相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。（补充：传统GBDT的实现也有学习速率）

- 剪枝
 当分裂时遇到一个负损失时，GBM会停止分裂。因此GBM实际上是一个贪心算法。
XGBoost会一直分裂到指定的最大深度(max_depth)，然后回过头来剪枝。如果某个节点之后不再有正值，它会去除这个分裂。
这种做法的优点，当一个负损失（如-2）后面有个正损失（如+10）的时候，就显现出来了。GBM会在-2处停下来，因为它遇到了一个负值。但是XGBoost会继续分裂，然后发现这两个分裂综合起来会得到+8，因此会保留这两个分裂。

- 缺失值处理 
对缺失值的处理。对于特征的值有缺失的样本，XGBoost内置处理缺失值的规则，可以自动学习出它的分裂方向。

- 并行
xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。

### 二、xgboost的参数意义？

1.通用参数：这些参数来控制XGBoost的宏观功能。

- booster [默认gbtree] ：选择每次迭代的模型，有两种选择：gbtree(基于树的模型) gbline(线性模型)
- silent [默认0]：当这个参数值为1时，静默模式开启，不会输出任何信息。一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。
- nthread [默认值为最大可能的线程数] 这个参数用来进行多线程控制，应当输入系统的核数。如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。

2.booster参数
尽管有两种booster可供选择，我这里只介绍tree booster，因为它的表现远远胜过linear booster，所以linear booster很少用到。
- eta [默认0.3] 和GBM中的 learning rate 参数类似。通过减少每一步的权重，可以提高模型的鲁棒性。典型值为0.01-0.2。
- min_child_weight [默认1]  决定最小叶子节点样本权重和。这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。这个参数需要用cv来调整。
- max_depth[默认6] 和GBM中的参数相同，这个值为树的最大深度。这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。需要使用CV函数来进行调优。典型值：3-10
- max_leaf_nodes 树上最大的节点或叶子的数量。可以替代max_depth的作用。因为如果生成的是二叉树，一个深度为n的树最多生成n2个叶子。如果定义了这个参数，GBM会忽略max_depth参数。
- gamma[默认0] 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
- max_delta_step[默认0] 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。这个参数一般用不到，但是你可以挖掘出来它更多的用处。
- subsample[默认1] 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。典型值：0.5-1
- colsample_bytree[默认1] 和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。典型值：0.5-1
- colsample_bylevel[默认1] 用来控制树的每一级的每一次分裂，对列数的采样的占比。我个人一般不太用这个参数，因为subsample参数和colsample_bytree参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处
- lambda[默认1]   权重的L2正则化项。(和Ridge regression类似)。这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。
- alpha[默认1] 权重的L1正则化项。(和Lasso regression类似)。可以应用在很高维度的情况下，使得算法的速度更快。
- scale_pos_weight[默认1] 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。

3.学习目标参数: 这些参数用来控制理想的优化目标和每一步结果的度量方法。
- objective[默认reg:linear] 这个参数定义需要被最小化的损失函数。最常用的值有：
    - binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
    - multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。 在这种情况下，你还需要多设一个参数：num_class(类别数目)。
    - multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
- eval_metric[默认值取决于objective参数的取值]
    - 对于有效数据的度量方法。
    - 对于回归问题，默认值是rmse，对于分类问题，默认值是error。
    - 典型值有： 
        rmse 均方根误差(∑Ni=1ϵ2N−−−−−√)
        mae 平均绝对误差(∑Ni=1|ϵ|N)
        logloss 负对数似然函数值
        error 二分类错误率(阈值为0.5)
        merror 多分类错误率
        mlogloss 多分类logloss损失函数
        auc 曲线下面积
- seed[默认0] 随机数的种子,设置它可以复现随机数据的结果，也可以用于调整参数


#### 推荐阅读
[XGBoost Parameters (official guide)](http://xgboost.readthedocs.org/en/latest/parameter.html#general-parameters)
[XGBoost Demo Codes (xgboost GitHub repository)](https://github.com/dmlc/xgboost/tree/master/demo/guide-python) 
[Python API Reference (official guide)](http://xgboost.readthedocs.org/en/latest/python/python_api.html)