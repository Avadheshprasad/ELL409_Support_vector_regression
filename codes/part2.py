

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

file_name='train_set.csv'
file_name1='test_set.csv'
data_x = np.genfromtxt(file_name, delimiter=',',usecols=[i for i in range(25)],encoding=None,dtype=float,skip_header=0)
data_y= np.genfromtxt(file_name, delimiter=',',usecols=25,encoding=None,dtype=int,skip_header=0)

pred_x= np.genfromtxt(file_name1, delimiter=',',usecols=[i for i in range(25)],encoding=None,dtype=float,skip_header=0)

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

def libsvm(x,y,c,ker):
    model=svm.SVC(kernel=ker,gamma=0.057)
    model.fit(x,y)  
    return model

m=libsvm(dx,dy,0.5,'rbf')
a=m.predict(pred_x)
print(m.score(dx,dy))
print(m.score(tx,ty))
print(m.get_params())
# for i in range(len(a)):
#     print(a[i])
print('end')

# def plot_vs_c(x,y,x1,y1):
#     c=np.zeros(20)
#     train_accuracy=np.zeros(20)
#     test_accuracy=np.zeros(20)
#     for i in range(20):
#         print(i)
#         c[i]=0.005*(i+1)
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

# def plot_vs_g(x,y,x1,y1):
#     c=np.zeros(20)
#     train_accuracy=np.zeros(20)
#     test_accuracy=np.zeros(20)
#     for i in range(20):
#         print(i)
#         c[i]=0.005*(i+1)
#         m=libsvm(x,y,c[i],'rbf')
#         test_accuracy[i]=m.score(x1,y1)
#         train_accuracy[i]=m.score(x,y)
#     plt.plot(c,train_accuracy,label = 'Training')
#     plt.plot(c,test_accuracy,label = 'Testing')
#     plt.xlabel('GAMMA')
#     plt.ylabel('ACCURACY')
#     plt.title('Accuracy Vs Gamma')

#     plt.legend()
#     plt.show()
    
# def plot_vs_d(x,y,x1,y1):
#     c=np.zeros(10)
#     train_accuracy=np.zeros(10)
#     test_accuracy=np.zeros(10)
#     for i in range(10):
#         print(i)
#         c[i]=(i+1)
#         m=libsvm(x,y,c[i],'poly')
#         test_accuracy[i]=m.score(x1,y1)
#         train_accuracy[i]=m.score(x,y)
#     plt.plot(c,train_accuracy,label = 'Training')
#     plt.plot(c,test_accuracy,label = 'Testing')
#     plt.xlabel('degree')
#     plt.ylabel('ACCURACY')
#     plt.title('Accuracy Vs degree')

#     plt.legend()
#     plt.show()    

# plot_vs_g(dx,dy,tx,ty)
with open('zoo.csv', 'w+') as f:
    f.write('Id,Class\n')
    for i in range(len(a)):
        if i + 1 < 1000: 
            f.write('{},{:d}\n'.format(str(i+1), int(a[i])))
        else:
            f.write('\"{:01d},{:03d}\",{:d}\n'.format((i+1)//1000, (i+1)%1000, int(a[i])))




