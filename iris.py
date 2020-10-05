# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import random


# 计算并返回均值向量
def junzhi(iris):
    a = np.zeros([4, 1])
    a[0] = np.mean(iris[:, 0])
    a[1] = np.mean(iris[:, 1])
    a[2] = np.mean(iris[:, 2])
    a[3] = np.mean(iris[:, 3])
    return a

# 计算类内离散度矩阵S_i
def S_i(iris):
    a = junzhi(iris)
    b = np.zeros([iris.shape[1], iris.shape[1]])
    for i in range(iris.shape[0]):
        b = b + np.matmul((iris[i, :].T - a), (iris[i, :].T - a).T)
    return b

# 计算类间离散度矩阵S_b
def S_b(iris1, iris2):
    b_1 = S_i(iris1)
    b_2 = S_i(iris2)
    c = b_1 + b_2
    return c

# 划分训练集、测试集的函数
def train_test(iris, target, num):
    train_iris, test_iris, train_target, test_target =\
        train_test_split(iris, target, test_size=0.6, random_state=num, shuffle=True)
    return {'train_iris': train_iris, 'test_iris': test_iris, 'train_target': train_target, 'test_target': test_target}

# 计算出最优的投影方向并计算出决策点
def best_w(iris1, iris2, target1, target2, num):
    train_iris1 = train_test(iris1, target1, num)['train_iris']
    train_iris2 = train_test(iris2, target2, num)['train_iris']
    s_0 = S_b(train_iris1, train_iris2)
    best_w = np.matmul(np.linalg.inv(s_0), junzhi(train_iris1) - junzhi(train_iris2))
    y_0 = 0.5*np.mean(np.matmul(train_iris1, best_w)) + 0.5*np.mean(np.matmul(train_iris2, best_w))
    # print(best_w)
    return best_w, y_0

# 对测试样本进行分类并计算相关评价指标
def classify(iris1,iris2,iris3,target1,target2,target3,num):
    # print(num)
    ## 训练集得到的最佳方向和决策点
    w_best12,y0_12=best_w(iris1, iris2, target1, target2, num)
    w_best13,y0_13=best_w(iris1, iris3, target1, target3, num)
    w_best23,y0_23=best_w(iris2, iris3, target2, target3, num)
    ## 测试集
    test1=train_test(iris1,target1,num)['test_iris']
    test2=train_test(iris2,target2,num)['test_iris']
    test3=train_test(iris3,target3,num)['test_iris']
    test=np.vstack((test1,test2,test3))
    ## 当前测试集对应的标签
    test_target1=train_test(iris1,target1,num)['test_target']
    test_target2=train_test(iris2,target2,num)['test_target']
    test_target3=train_test(iris3,target3,num)['test_target']
    test_target=np.hstack((test_target1,test_target2,test_target3))
    # print(test_target)
    ## 存放预测得到的标签
    predict_target=np.zeros_like(test_target)
    ## 先通过第一二类决策函数
    y=np.matmul(test,w_best12)
    for i in range(len(test)):
        if y[i]>y0_12 or y[i]==y0_12:
            predict_target[i]=0
        else:
            predict_target[i]=1
    ## 再通过第一三类决策函数
    y=np.matmul(test,w_best13)
    for i in range(len(test)):
        if y[i]>y0_13 or y[i]==y0_13:
            predict_target[i]=0
        else:
            predict_target[i]=2

    ## 剩余的通过第二三类决策函数
    y=np.matmul(test,w_best23)
    for i in range(len(test)):
        if predict_target[i]!=0:
            if y[i]>y0_23 or y[i]==y0_23:
                predict_target[i]=1
            else:
                predict_target[i]=2
    # print(predict_target)
    ## 计算OA、AA、kappa系数
    ### 记录三类样本分类正确的数量
    num_1=0
    num_2=0
    num_3=0
    ### 记录三类样本实际分类的数量
    real_num_1=0
    real_num_2=0
    real_num_3=0 
    for i in range(len(test_target)):
        ## 统计分类正确的数量
        if i<len(test_target)/3:
            if predict_target[i]==test_target[i]:
                num_1=num_1+1
        elif i<2*(len(test_target))/3:
            if predict_target[i]==test_target[i]:
                num_2=num_2+1
        else:
            if predict_target[i]==test_target[i]:
                num_3=num_3+1
        ## 统计实际的分类数量
        if predict_target[i]==0:
            real_num_1=real_num_1+1
        elif predict_target[i]==1:
            real_num_2=real_num_2+1
        else:
            real_num_3=real_num_3+1
    # print(num_1)
    ### 计算相关指标
    OA=(num_1+num_2+num_3)/len(test_target)
    AA_1=num_1*3/len(test_target)
    AA_2=num_2*3/len(test_target)
    AA_3=num_3*3/len(test_target)
    pe=(real_num_1*len(test_target1)+real_num_2*len(test_target2)\
        +real_num_3*len(test_target3))/np.square(len(test_target))
    kappa=(OA-pe)/(1-pe)
    return OA,[AA_1,AA_2,AA_3],kappa


if __name__ == "__main__":
    # 导入数据并去量纲
    data = load_iris()
    iris1 = data.data[0:50, 0:4]
    iris2 = data.data[50:100, 0:4]
    iris3 = data.data[100:150, 0:4]
    
    max_feature=np.array([np.max(data.data[:,0]),np.max(data.data[:,1]),\
        np.max(data.data[:,2]),np.max(data.data[:,3])])
    iris1=iris1/max_feature
    iris2=iris2/max_feature
    iris3=iris3/max_feature
    # 导入标签
    target1 = data.target[0:50].T
    target2 = data.target[50:100].T
    target3 = data.target[100:150].T

    # 存储相关的指标值
    OAs=np.zeros([30,1])
    AAs=np.zeros([30,3])
    kappas=np.zeros([30,1])

    #随机给出30个随机种子，用于train_test函数
    # nums=[703,5205,8248,4998,1027,8528,7063,6513,793,2805,1524,8985,3939,9000\
    # ,3796,3178,628,9359,582,265,5920,8866,7960,5090,5481,4928,526,8763,5333,6596]
    nums=random.sample(range(0,10000),100)
    j=0;k=0
    while j<30:
        try:
            OAs[j],AAs[j,],kappas[j]=classify(iris1,iris2,iris3,target1,target2,target3,nums[k])
            j=j+1
            k=k+1
        except:
            k+=1
    # print(OAs)
    temp=0
    for i in range(len(OAs)):
        if OAs[i]!=0:
            temp=temp+1
    # print(AAs)
    # print(OAs)
    # print(kappas)
    print("AA值分别为：\n{:.3f}\n{:.3f}\n{:.3f}".format(np.sum(AAs[:,0])/temp,np.sum(AAs[:,1]/temp),np.sum(AAs[:,2])/temp))
    print("OA值为：\n{:.3f}".format(np.sum(OAs)/temp))
    print("kappa值为：\n{:.3f}".format(np.sum(kappas)/temp))












