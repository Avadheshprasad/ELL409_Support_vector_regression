
import numpy as np
import pandas as pd

file_name='2019MT60747.csv'
datafile= pd.read_csv(file_name,header=None,nrows=3000)
data=np.array((datafile.sort_values(datafile.columns[25])).values)

y=data[100:400,25].astype(int)
x=data[100:400,:10]

for i in range(len(y)):
    if(y[i]==0):
        y[i]=1
    else:
        y[i]=-1

def L_svm(a_i,a_j,c,yi,yj):
    if(yi!=yj):
        return min(c,c+a_j-a_i)
    else:
        return min(c,a_i+a_j)

def H_svm(a_i,a_j,c,yi,yj):
    if(yi!=yj):
        return max(0,a_j-a_i)
    else:
        return max(0,a_i+a_j-c)
    
def linear_kernel(x1,x2):
    return np.dot(x1,x2)
def polynomial_kernel(x,y,gamma,d):
    return (1+gamma*np.dot(x,y))**d

def E_i(a,b,x,y,i,m):
    f=b
    for j in range(m):
        f+=a[j]*y[j]*(np.inner(x[j],x[i]))
    return f-y[i]


def takeStep(i1,i2,point,target,b,eps,alpha,C):
    if (i1 == i2):
            return 0
    m=len(target)
    alph1=alpha[i1]
    alph2=alpha[i2]
    y1 = target[i1]
    y2= target[i2]
    E1 = E_i(alpha,b,point,target,i1,m)
    E2= E_i(alpha,b,point,target,i2,m)
    s = y1*y2
    L=L_svm(alph1,alph2,C,target[i1],target[i2])
    H=H_svm(alph1,alph2,C,target[i1],target[i2])
    if (L == H):
        return 0
    k11 = linear_kernel(point[i1],point[i1])
    k12 = linear_kernel(point[i1],point[i2])
    k22 = linear_kernel(point[i2],point[i2])
    eta = k11+k22-2*k12
    if (eta > 0):
        a2 = alph2 + y2*(E1-E2)/eta
        if (a2 < L):
            a2 = L
        elif (a2 > H):
            a2 = H
    else:
        f1=y1*(E1+b)-alph1*k11-s*alph2*k12
        f2=y2*(E2+b)-s*alph1*k12-alph2*k22
        L1=alph1+s*(alph2-L)
        H1=alph1+s*(alph2-H)
        Lobj = L1*f1+L*f2+0.5*L1*L1*K11+0.5*L*L*K22+S*L*L1*K12
        Hobj = H1*f1+H*f2+0.5*H1*H1*K11+0.5*H*H*K22+S*H*H1*K12
        if (Lobj < Hobj-eps):
            a2 = L
        elif (Lobj > Hobj+eps):
            a2 = H
        else:
            a2 = alph2
    if (np.abs(a2-alph2) < eps*(a2+alph2+eps)):
        return 0
    a1 = alph1+s*(alph2-a2)
    b1=b-E1-target[i1]*(a1-alph1)*(linear_kernel(point[i1],point[i1]))-target[i2]*(a2-alph2)*(linear_kernel(point[i1],point[i2]))
    b2=b-E2-target[i1]*(a1-alph1)*(linear_kernel(point[i1],point[i2]))-target[i2]*(a2-alph2)*(linear_kernel(point[i2],point[i2]))

    if(0<a1<C):
        b=b1
    elif(0<a2<C):
        b=b2
    else:
        b=(b1+b2)/2
    alpha[i1]=a1
    alpha[i2]=a2
    return 1

def non_zero_non_c_alpha(alpha,C):
    res=0
    for i in range(len(alpha)):
        if(alpha[i]<C and alpha[i]>0):
            res+=1
    return res


def examineExample(i2,point,target,b,alpha,tol,C,eps):
    m=len(alpha)
    y2 = target[i2]
    alph2 = alpha[i2]
    E2 = E_i(alpha,b,point,target,i2,m)
    r2 = E2*y2
    if ((r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0)):
        z= non_zero_non_c_alpha(alpha,C)
        if (z > 1):
            pass
#             i1 = result of second choice heuristic (section 2.2)
            if takeStep(i1,i2,point,target,b,eps,alpha,C):
                return 1
        for i in range(len(alpha)):
            if(alpha[i]>0 or alpha[i]<C):
                if takeStep(i,i2,point,target,b,eps,alpha,C):
                    return 1
        for i1 in range(len(alpha)):
            if (takeStep(i1,i2,point,target,b,eps,alpha,C)):
                return 1
    
    return 0

def main_routine(x,y,eps,tol,C):
    numChanged = 0
    examineAll = 1
    alpha=np.zeros(len(y))
    b=0
    while (numChanged > 0 or examineAll):
        numChanged = 0
        if (examineAll):
            for i in range(len(y)):
                numChanged += examineExample(i,x,y,b,alpha,tol,C,eps)
        else:
            for i in range(len(alpha)):
                if(alpha[i]>0 or alpha[i]<C):
                    numChanged += examineExample(i,x,y,b,alpha,tol,C,eps)
        if (examineAll == 1):
            examineAll = 0
        elif (numChanged == 0):
            examineAll = 1
    return alpha,b

a,b=main_routine(x,y,0.001,0.001,1)
print(a)



