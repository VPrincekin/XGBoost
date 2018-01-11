#!/usr/bin/python
#coding=utf-8
import numpy as np
import xgboost as xgb
import pickle
import scipy.sparse
"""
一个简单的例子
"""
#从文本文件加载文件，也是由xgboost生成的二进制缓冲区
dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')

#通过map指定参数
param = {'max_depth':2,'eta':1,'silent':0,'objective':'binary:logistic'}

#在训练期间要评估的项目列表，这允许用户观看在验证集上的性能。
watchlist = [(dtest,'eval'),(dtrain,'train')]
#迭代次数
num_round = 4
#训练
bst = xgb.train(param,dtrain,num_round,watchlist)
#预测
preds = bst.predict(dtest)
print(preds)
labels = dtest.get_label()
print(labels)
#计算错误率
print('error=%f' %(sum(1 for i in range(len(preds))if int(preds[i]>0.5) != labels[i])/float(len(preds))))

"""
模型的复用
"""
#将数据保存到二进制缓冲区中
dtest.save_binary('dtest.buffer')
#转储模型与功能图
bst.dump_model('dump.nice.txt', '../data/featmap.txt')
#保存模型
bst.save_model('xgb.model')
#加载模型和数据
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
#直接预测
preds2 = bst2.predict(dtest2)
#断言它们是一样的
assert np.sum(np.abs(preds2-preds))==0


#或者，你可以pickle这个booster
pks = pickle.dumps(bst2)
#加载模型和数据
bst3 = pickle.loads(pks)
preds3 = bst3.predict(dtest2)
#断言它们是一样的
assert np.sum(np.abs(preds3-preds))==0

"""
从scipy.sparse构建dmatrix
"""
labels=[];row=[];col=[];dat=[]
i=0
for line in open('../data/agaricus.txt.train').readlines():
    arr = line.split()
    labels.append(int(arr[0]))
    for it in arr[1:]:
        k,v = it.split(':')
        dat.append(int(v))
        row.append(i)
        col.append(int(k))
    i+=1
csr = scipy.sparse.csc_matrix((dat,(row,col)))
"""
也可以从numpy array构建
npymat = csr.todense()
dtrain = xgb.DMatrix(npymat, label=labels)
"""
dtrain = xgb.DMatrix(csr,label=labels)
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)

