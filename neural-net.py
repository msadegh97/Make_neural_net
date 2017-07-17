import numpy as np
from math import exp


ninput=np.array([[6,2,1]
                ,[1,4,0]
                ,[4,8,5]
                ,[1,1,2]])
noutput=np.array([[.7],[1],[1],[0]])
synops0=2*np.random.random((3,7))-1
synops1=2*np.random.random((7,1))-1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivetive(x):
    return x*(1-x)
for i in range(10000):

    l0=ninput
    l1=sigmoid(np.dot(l0,synops0))
    l2=sigmoid(np.dot(l1,synops1))

    l2_error=noutput-l2
    if i%1000==0:
        print("Error:",np.mean(abs(l2_error)))
    l2_delta=l2_error*derivetive(l2)
    l1_error=l2_delta.dot(synops1.T)
    l1_delta=l1_error*derivetive(l1)


    synops1 += l1.T.dot(l2_delta)
    synops0 += l0.T.dot(l1_delta)

print ("ountput: ",l2)
