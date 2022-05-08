import numpy as np
import cvxopt
import cvxopt.solvers
import pandas as pd
from cvxopt import matrix,solvers
from numpy import linalg
import time



file_name='2019MT60747.csv'
data_x = np.genfromtxt(file_name, delimiter=',',usecols=[i for i in range(25)],encoding=None,dtype=None,skip_header=0)
data_y= np.genfromtxt(file_name, delimiter=',',usecols=25,encoding=None,dtype=int,skip_header=0)

n=len(data_y)
a1,b1,a2,b2,a3,b3=1,9,4,5,3,8
x1,x2,x3,y1,y2,y3=[],[],[],[],[],[]

for i in range(n):
    if (data_y[i]==a1 or data_y[i]==b1):
        x1.append(data_x[i])
        y1.append(data_y[i])
    elif(data_y[i]==a2 or data_y[i]==b2):
        x2.append(data_x[i])
        y2.append(data_y[i])
    elif(data_y[i]==a3 or data_y[i]==b3):
        x3.append(data_x[i])
        y3.append(data_y[i])
        
def training_testing_data(x,y,f,features):
    n1=int(f*len(y))
    x=[M[:features] for M in x]
    train_x=x[:n1]
    train_y=y[:n1]
    test_x=x[n1:]
    test_y=y[n1:]
    return train_x,train_y,test_x,test_y

#training_data and testing_data
f=0.75
dx,dy,tx,ty=training_testing_data(data_x,data_y,f,25)
dx1,dy1,tx1,ty1=training_testing_data(x1,y1,f,10)
dx2,dy2,tx2,ty2=training_testing_data(x2,y2,f,10)
dx3,dy3,tx3,ty3=training_testing_data(x3,y3,f,10)

        
def convert(y,dy):
    a=y[0]
    for i in range(len(y)):
        if(y[i]==a):
            y[i]=1
        else:
            y[i]=-1
    for i in range(len(dy)):
        if(dy[i]==a):
            dy[i]=1
        else:
            dy[i]=-1
    return y,dy


def linear_kernel(x1,x2,gamma,d):
    return np.dot(x1,x2)
def polynomial_kernel(x,y,gamma,d):
    return (1+gamma*np.dot(x,y))**d
def radial_kernel(x,y,gamma,d):
    return np.exp(-gamma*linalg.norm(x-y)**2)


def fit(C,X, y,gamma,d,kernel):
    m=len(y)
    K=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
                K[i,j]= kernel(X[i],X[j],gamma,d)*1.
    P = matrix(np.outer(y,y)*K)
    q = matrix(np.ones(m)*-1)
    y1=y
    for i in range(m):
        y1[i]=y1[i]*1.
    A = matrix(y1,(1,m)) 
    b = matrix(np.zeros(1))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    Ans = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(Ans['x'])
    return alphas


def accuracy(x,y,w,b):
    p=np.matmul(x,w)+b
    m=len(y)
    for j in range(m):
        if(p[j]>=0):
            p[j]=1
        else:
            p[j]=-1
    match=0
    for i in range(m):
        if(p[i]*y[i]>0):
             match+=1
    return match/m

def find_w(alpha,x,y,features):
    w=np.zeros(features)
    for i in range(len(alpha)):
        w+=x[i]*alpha[i]*y[i]
    return w

def print_support_vector(alpha,limit):
    print('support_vector=',np.where(alpha > limit)[0])

def result(x,y,dx,dy):
    y,dy=convert(y,dy)
    
    start = time.time()
    alpha=fit(1,x,y,0.001,3,linear_kernel)
    end = time.time()
    print('time=',end - start)
    support_vectors = print_support_vector(alpha,1e-4)
    w=find_w(alpha,x,y,len(x[0]))
    print('accuracy=',accuracy(dx,dy,w,1))
    print('')
    
result(dx1,dy1,tx1,ty1)
result(dx2,dy2,tx2,ty2)




