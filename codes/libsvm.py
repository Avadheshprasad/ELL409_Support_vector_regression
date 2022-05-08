

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time
style.use("ggplot")

file_name='2019MT60747.csv'
data_x = np.genfromtxt(file_name, delimiter=',',usecols=[i for i in range(25)],encoding=None,dtype=float,skip_header=0)
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
dx,dy,tx,ty=training_testing_data(data_x,data_y,f,10)
dx1,dy1,tx1,ty1=training_testing_data(x1,y1,f,10)
dx2,dy2,tx2,ty2=training_testing_data(x2,y2,f,10)
dx3,dy3,tx3,ty3=training_testing_data(x3,y3,f,10)



def libsvm(x,y,c,ker,g):
    model=svm.SVC(kernel=ker,C=c,gamma=g)
    model.fit(x,y)  
    return model

start=time.time()
m=libsvm(dx1,dy1,1.0,'poly',0.05)
print(m.predict(dx1))
end = time.time()
print('time=',end - start)
print('accuracy=',m.score(tx1,ty1))
print(m.get_params())

# # linear kernel tune c
# def plot_vs_c(x,y,x1,y1):
#     c=np.zeros(100)
#     train_accuracy=np.zeros(100)
#     test_accuracy=np.zeros(100)
#     for i in range(100):
#         c[i]=0.001*(i+1)
#         m=libsvm(x,y,c[i],'linear')
#         test_accuracy[i]=m.score(x1,y1)
#         train_accuracy[i]=m.score(x,y)
#     plt.plot(c,train_accuracy,label = 'Training')
#     plt.plot(c,test_accuracy,label = 'Testing')
#     plt.xlabel('SOFT MARGIN COST(C)')
#     plt.ylabel('ACCURACY')
#     plt.title('Accuracy Vs C')

#     plt.legend()
#     plt.show()

# plot_vs_c(dx1,dy1,tx1,ty1)
# plot_vs_c(dx2,dy2,tx2,ty2)
# plot_vs_c(dx3,dy3,tx3,ty3)

# polynomial kernel tune c

# def plot_vs_c1(x,y,x1,y1):
#     c=np.zeros(200)
#     train_accuracy=np.zeros(200)
#     test_accuracy=np.zeros(200)
#     for i in range(200):
#         c[i]=0.005*(i+1)
#         m=libsvm(x,y,c[i],'poly')
#         test_accuracy[i]=m.score(x1,y1)
#         train_accuracy[i]=m.score(x,y)
#     plt.plot(c,train_accuracy,label = 'Training')
#     plt.plot(c,test_accuracy,label = 'Testing')
#     plt.xlabel('SOFT MARGIN COST(C)')
#     plt.ylabel('ACCURACY')
#     plt.title('Accuracy Vs C')

#     plt.legend()
#     plt.show()
# print(2)
# plot_vs_c1(dx1,dy1,tx1,ty1)
# plot_vs_c1(dx2,dy2,tx2,ty2)
# plot_vs_c1(dx3,dy3,tx3,ty3)



a=(svm.SVC(kernel = 'linear', C = 1)).fit(dx1,dy1).support_
# a1=(svm.SVC(kernel = 'linear', C = 1)).fit(dx2,dy2).support_
# a2=(svm.SVC(kernel = 'linear', C = 1)).fit(dx3,dy3).support_
a.sort()
# b.sort()
print(a)
# print(a1)
# print(a2)


# rbf kernel
# def plot_vs_c2(x,y,x1,y1):
#     c=np.zeros(400)
#     train_accuracy=np.zeros(400)
#     test_accuracy=np.zeros(400)
#     for i in range(400):
#         c[i]=0.01*(i+1)
#         m=libsvm(x,y,c[i],'rbf')
#         test_accuracy[i]=m.score(x1,y1)
#         train_accuracy[i]=m.score(x,y)
#     plt.plot(c,train_accuracy,label = 'Training')
#     plt.plot(c,test_accuracy,label = 'Testing')
#     plt.xlabel('SOFT MARGIN COST(C)')
#     plt.ylabel('ACCURACY')
#     plt.title('Accuracy Vs C')

#     plt.legend()
#     plt.show()




