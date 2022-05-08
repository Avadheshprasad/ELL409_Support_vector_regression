

import numpy as np
import time
import pandas as pd
import random


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


def E_i(a,b,x,y,i,m,kernel,gamma,d):
    f=b
    for j in range(m):
        f+=a[j]*y[j]*(kernel(x[j],x[i],gamma,d))
    return f-y[i]

def H_svm(a_i,a_j,c,yi,yj):
    if(yi!=yj):
        return min(c,c+a_j-a_i)
    else:
        return min(c,a_i+a_j)

def L_svm(a_i,a_j,c,yi,yj):
    if(yi!=yj):
        return max(0,a_j-a_i)
    else:
        return max(0,a_i+a_j-c)

def eta(x,i,j,kernel,gamma,d):
    return 2*(kernel(x[i],x[j],gamma,d))-kernel(x[i],x[i],gamma,d)-kernel(x[j],x[j],gamma,d)

def linear_kernel(x1,x2,gamma,d):
    return np.dot(x1,x2)
def polynomial_kernel(x,y,gamma,d):
    return (1+gamma*np.dot(x,y))**d
def radial_kernel(x,y,gamma,d):
    return np.exp(-gamma*linalg.norm(x-y)**2)

def simplified_smo(C,tol,max_passes,dx,dy,kernel,gamma,d):
    m=len(dy)
    a=np.zeros(m)
    b=0
    passes=0
    while(passes<max_passes):
        num_changed_alphas=0
        for i in range(m):
            Ei=E_i(a,b,dx,dy,i,m,kernel,gamma,d)
            if((dy[i]*Ei< -tol and a[i]<C) or (dy[i]*Ei>tol and a[i]>0)):
                num=*range(0,i),*range(i+1,m)
                j=random.choice(num)
                Ej=E_i(a,b,dx,dy,j,m,kernel,gamma,d)
                ai_old=a[i]
                aj_old=a[j]
                L=L_svm(ai_old,aj_old,C,dy[i],dy[j])
                H=H_svm(ai_old,aj_old,C,dy[i],dy[j])
                if(L==H):
                    continue
                n=eta(dx,i,j,kernel,gamma,d)
                if(n>=0):
                    continue
                a[j]=a[j]- (dy[j]*(Ei-Ej))/n
                if(a[j]>H):
                    a[j]=H
                if(a[j]<L):
                    a[j]=L
                
                if(np.abs(a[j]-aj_old)<tol):
                    continue
                a[i]=a[i]+dy[i]*dy[j]*(aj_old-a[j])
                
                b1=b-Ei-dy[i]*(a[i]-ai_old)*(kernel(dx[i],dx[i],gamma,d))-dy[j]*(a[j]-aj_old)*(kernel(dx[i],dx[j],gamma,d))
                b2=b-Ej-dy[i]*(a[i]-ai_old)*(kernel(dx[i],dx[j],gamma,d))-dy[j]*(a[j]-aj_old)*(kernel(dx[j],dx[j],gamma,d))
                
                if(0<a[i]<C):
                    b=b1
                elif(0<a[j]<C):
                    b=b2
                else:
                    b=(b1+b2)/2
                num_changed_alphas+=1
        if(num_changed_alphas==0):
            passes+=1
        else:
            passes=0
    return a,b



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


def result(x,y,dx,dy,kernel,c,gamma):
    y,dy=convert(y,dy)
    start = time.time()
    alpha,b=simplified_smo(c,0.005,1,x,y,kernel,gamma,3)
    end = time.time()
    print('time=',end - start)
#     print(alpha)
    w=find_w(alpha,x,y,len(x[0]))
    print('w=',w)
    print('accuracy=',accuracy(dx,dy,w,b))
    
    
result(dx1,dy1,tx1,ty1,linear_kernel,0.01,'scale')





