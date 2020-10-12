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

# 计算总类间离散度矩阵S_w
def S_w(iris1, iris2, iris3):
    b_1 = S_i(iris1)
    b_2 = S_i(iris2)
    b_3 = S_i(iris3)
    c = b_1 + b_2 + b_3
    return c

# 计算样本类间离散度矩阵S_b
def S_b(iris1,iris2,iris3):
    # 求类别均值以及所有样本的均值
    mu_iris1 = junzhi(iris1)
    mu_iris2 = junzhi(iris2)
    mu_iris3 = junzhi(iris3)
    mu = (mu_iris1+mu_iris2+mu_iris3)/3
    c=iris1.shape[0]*np.matmul(mu_iris1-mu,(mu_iris1-mu).T)+ \
      iris2.shape[0]*np.matmul(mu_iris2-mu,(mu_iris2-mu).T)+ \
      iris3.shape[0]*np.matmul(mu_iris3-mu,(mu_iris3-mu).T)
    return c

# 划分训练集、测试集的函数
def train_test(iris, target, num):
    train_iris, test_iris, train_target, test_target = \
        train_test_split(iris, target, test_size=0.6, random_state=num, shuffle=True)
    return {'train_iris': train_iris, 'test_iris': test_iris, 'train_target': train_target, 'test_target': test_target}

# 计算出最优的投影方向(到2维平面)
def best_w(iris1,iris2,iris3,target1,target2,target3,num):
    train_iris1 = train_test(iris1, target1, num)['train_iris']
    train_iris2 = train_test(iris2, target2, num)['train_iris']
    train_iris3 = train_test(iris3, target3, num)['train_iris']
    s_b = S_b(train_iris1, train_iris2, train_iris3)
    s_w = S_w(train_iris1, train_iris2, train_iris3)
    eigvalue,eigvector=np.linalg.eig(np.matmul(np.linalg.inv(s_w),s_b))
    eig=np.vstack((eigvalue,eigvector))
    eig=eig[:,eig[0].argsort()]
    best_w=eig[1:,-2:]
    return best_w

# 根据k值计算点，并返回类别
def knn(x,y,test,test_target,k):
    dis=np.zeros([test.shape[0],1])
    for i in range(test.shape[0]):
        dis[i,0]=np.sqrt((x.real-test[i,0].real)**2+(y.real-test[i,1].real)**2)

    a=np.column_stack((test,dis))
    # print(a.shape)
    a=np.column_stack((a,test_target))
    a=a[a[:,2].argsort()]
    # print(a)
    a=a[1:k+1,:]
    ## 设置一个数组来统计各类的个数
    n=np.zeros([3,1])
    for i in range(a.shape[0]):
        if a[i,3]==0:
            n[0]+=1
        elif a[i,3]==1:
            n[1]+=1
        else:
            n[2]+=1
    # print(n)
    return np.where(n==np.max(n))[0][0]

# 给定参数，画出分类结果图
def ploty(test,predict_target,real_num_1,real_num_2,real_num_3):
    a=np.column_stack((test,predict_target))
    a=a[a[:,2].argsort()]
    print(a)
    plt.figure(1)
    plt.scatter(a[0:real_num_1,0],a[0:real_num_1,1],color='red')
    plt.scatter(a[real_num_1:real_num_1+real_num_2,0],a[real_num_1:real_num_1+real_num_2,0],color='blue',marker='x')
    # plt.scatter(a[real_num_2:real_num_2+real_num_3,0],a[real_num_2:real_num_2+real_num_3,0],color='green',marker='o')
    plt.show()

# 对测试样本进行分类并计算相关评价指标
def classify(iris1,iris2,iris3,target1,target2,target3,num,k=5):
    # print(num)
    ## 训练集得到的投影方向
    w_best=best_w(iris1,iris2,iris3,target1,target2,target3,num)
    ## 测试集
    test1_iris=train_test(iris1,target1,num)['test_iris']
    test2_iris=train_test(iris2,target2,num)['test_iris']
    test3_iris=train_test(iris3,target3,num)['test_iris']
    test_iris=np.vstack((test1_iris,test2_iris,test3_iris))
    ## 将测试集按照投影方向投影
    test=np.matmul(test_iris,w_best)
    # print(test.shape)
    ## 当前测试集对应的标签
    test_target1=train_test(iris1,target1,num)['test_target']
    test_target2=train_test(iris2,target2,num)['test_target']
    test_target3=train_test(iris3,target3,num)['test_target']
    test_target=np.hstack((test_target1,test_target2,test_target3))
    # print(test_target.shape)
    ## 存放预测得到的标签
    predict_target=np.zeros_like(test_target)
    # 利用knn进行分类
    for i in range(test.shape[0]):
        predict_target[i]=knn(test[i,0],test[i,1],test,test_target,k)
    # print(predict_target.shape)
    # print(test_target)
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
    pe=(real_num_1*len(test_target1)+real_num_2*len(test_target2) \
        +real_num_3*len(test_target3))/np.square(len(test_target))
    kappa=(OA-pe)/(1-pe)
    # 结果图
    # ploty(test,predict_target,real_num_1,real_num_2,real_num_3)
    return OA,[AA_1,AA_2,AA_3],kappa


if __name__ == "__main__":
    # 导入数据并去量纲
    data = load_iris()
    iris1 = data.data[0:50, 0:4]
    iris2 = data.data[50:100, 0:4]
    iris3 = data.data[100:150, 0:4]

    max_feature=np.array([np.max(data.data[:,0]),np.max(data.data[:,1]), \
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
    nums=[7136,2619,4794,5676,3265,1979,906,9058,2011,5435,1729,4558,8765,6826,6174,6286,7825,
          4843,9697,788,8082,3785,8206,5383,4723,7249,4929,4722,9587,4400]
    # nums=random.sample(range(0,10000),100)
    j=0;i=0
    while j<30:
        try:
            OAs[j],AAs[j,],kappas[j]=classify(iris1,iris2,iris3,target1,target2,target3,nums[i])
            j=j+1
            i=i+1
        except:
            i+=1
    #     print(j)
    # OAs[j],AAs[j,],kappas[j]=classify(iris1,iris2,iris3,target1,target2,target3,9587)
    temp=0
    for i in range(len(OAs)):
        if OAs[i]!=0:
            temp=temp+1
    # print(np.around(AAs,decimals=3))
    # print(np.around(OAs,decimals=3))
    # print(np.around(kappas,decimals=3))
    print("AA值分别为：\n{:.3f}\n{:.3f}\n{:.3f}".format(np.sum(AAs[:,0])/temp,np.sum(AAs[:,1]/temp),np.sum(AAs[:,2])/temp))
    print("OA值为：\n{:.3f}".format(np.sum(OAs)/temp))
    print("kappa值为：\n{:.3f}".format(np.sum(kappas)/temp))












