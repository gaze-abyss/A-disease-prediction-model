#%%
# 导入依赖库
import pandas as pnd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import operator
from sklearn.model_selection import KFold
 
import itertools
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
 
from sklearn import tree
import seaborn as sns
 
from IPython.display import Image
#%%
#读取数据
#%%
rawdata = pnd.read_csv('C:/Users/hjwang/Desktop/data2.csv')
#%%
#填补缺失值
for c in rawdata.columns[:-1]:
    rawdata[c] = rawdata[c].apply(lambda x:rawdata[rawdata[c]!=9][c].astype(float).mean() if x == 9 else x)
    rawdata[c] = rawdata[c].astype(float)
#%%
#数据预处理
# if "diagnosis" == 0, 没有疾病
# if "diagnosis" >= 1, 有疾病
rawdata.loc[:, "diag_int"] = rawdata.loc[:, "sample"].apply(lambda x: 1 if x >= 1 else 0)
preprocessing.Normalizer().fit_transform(rawdata)

#数据标准化
preprocessing.Normalizer().fit_transform(rawdata)
#%%
#划分数据集
rawdata_train, rawdata_test, goal_train, goal_test = model_selection.train_test_split(rawdata.loc[:,'rs10465729':'rs11577368'], rawdata.loc[:,'diag_int'], test_size=0.33, random_state=0) 

#计算相关系数
corr = rawdata.corr()
#%%
# 添加参数
loss = ["hinge", "log"]
penalty = ["l1", "l2"]
alpha = [0.1, 0.05, 0.01]
n_iter = [500, 1000]

# 用不同的参数建立模型，选择最佳的组合以获得最高的精度
best_score = 0
best_param = (0,0,0,0)
for l in loss:
    for p in penalty:
        for a in alpha:
            for n in n_iter:
                print("Parameters for model", (l,p,a,n))
                lss = SGDClassifier(loss=l, penalty=p, alpha=a, n_iter_no_change=n,max_iter=10000)
                lss.fit(rawdata_train, goal_train)
                print("Linear regression SGD Cross-Validation scores:")
                scores = model_selection.cross_val_score(lss, rawdata.loc[:,'rs10465729':'rs11577368'], rawdata.loc[:,'diag_int'], cv=10)
                print(scores)
                print("Mean Linear regression SGD Cross-Validation score = ", np.mean(scores))
                
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_param = (l,p,a,n)
                    
    
print("The best parameters for model are ", best_param)
print("The Cross-Validation score = ", best_score)
#%%
#建立了具有最佳参数的模型
lss_best = SGDClassifier(alpha=0.05, fit_intercept=True, loss='log', n_iter_no_change=1000, penalty='l2')
lss_best.fit(rawdata_train, goal_train)
print("Linear regression SGD Test score:")
print(lss_best.score(rawdata_test, goal_test))

#计算混合矩阵
cnf_matrix = confusion_matrix(goal_test, lss_best.predict(rawdata_test))
np.set_printoptions(precision=2)


#获取性能指标
scores = ['accuracy', 'f1', 'precision', 'recall']

metrics = {score: model_selection.cross_val_score(lss_best,rawdata_test, goal_test, scoring=score).mean() for score in scores}

metrics

#%%
# 为测试集进行预测
y_pred_proba = lss_best.predict_proba(rawdata_test)
y_pred_proba = [x[1] for x in y_pred_proba]
for i in y_pred_proba[:10]:
    print (i)

y_pred = lss_best.predict(rawdata_test)

test_df = pnd.DataFrame(rawdata_test)

test_df.loc[:, "Disease_probability"] = [x[1] for x in lss_best.predict_proba(rawdata_test)]
test_df.to_excel("disease_probability.xlsx", index = False)
test_df[:]
test_df.to_csv('C:/Users/hjwang/Desktop/result2.all.csv')
#%%
#从构建的LSS模型中计算每个SNP的权重
w = lss_best.coef_[0]
a = -w[0] / w[1]
print ("Weight Coefficients")
coeff_df = pnd.DataFrame(columns = ['X_k', 'coeff'])
for c in xrange(len(rawdata.loc[:,'rs12611091':'rs104894124'].columns)):
    coeff_df.loc[len(coeff_df)] = [rawdata.loc[:,'0':'33'].columns[c], w[c]]
    
coeff_df
# %%
