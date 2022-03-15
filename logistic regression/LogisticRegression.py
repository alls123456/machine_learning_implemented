import pandas as pd
import numpy as np
import time
import random

def loadData(filename):
    data = pd.read_csv(filename, header=None)

    #给所有数据增加一项常数项
    data[785] = 1;
    data = data.values

    #data第0列为标签，1-785为特征
    y_label = data[:, 0]
    X_label = data[:, 1:]

    #使用二分类逻辑回归，将数据以零为界分为两类
    y_label[y_label>0] = 1
    return X_label, y_label

def sigmoid(X):
    return 1 / (1+np.exp(-1*X))

def logisticRegression(X_train, y_train, epochs):
    w = np.mat([random.uniform(0,1) for _ in range(len(X_train[0]))]).reshape(-1, 1)
    X_train = np.mat(X_train)
    y_train = np.mat(y_train)

    #训练
    print("start to train")

    learning_rate = 0.001
    for i in range(epochs):
        hx = sigmoid(X_train @ w)

        print(f'in {i} epoch')
        w -= learning_rate*X_train.T@(hx-y_train.T)
    return w

def predict(x, w):
    hx = sigmoid(x@w)
    if hx >= 0.5:
        return 1
    if hx < 0.5:
        return 0


def test(X_test, y_test, w):
    acc = 0
    acc_num = 0
    for i in range(len(X_test)):
        x = np.mat(X_test[i])
        y_pred = predict(x, w)
        if y_pred == y_test[i]:
            acc_num += 1
        print(f'find {i}th data cluster: y_pred={y_pred}, y={y_test[i]}')
        print('now_acc', acc_num / (i+1))

if __name__ == "__main__":
    #获取当前时间
    start = time.time()

    #读取训练文件
    print('load TrainData')
    X_train, y_train = loadData('../Mnist/mnist_train.csv')

    #读取测试文件
    print('load TestData')
    X_test, y_test = loadData('../Mnist/mnist_test.csv')

    #开始训练，得到系数
    w = logisticRegression(X_train, y_train, 200)
    test(X_test, y_test, w)

    #获取结束时间
    end = time.time()

    print('run time', end - start)