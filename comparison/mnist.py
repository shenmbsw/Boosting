import scipy.io
import numpy as np
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata


mnist = fetch_mldata('MNIST original', data_home="data/mnist")

X,Y = shuffle(mnist.data,mnist.target,random_state=0)
print(X)
print(Y)

# Salt-and-pepper noise
rand = len(X)
print(rand)

testlen=round(0.7*len(Y))
scores=list()
for j in range(10,100,10):
    score = 0
    i = 20;
    XNoise = X;
    indices = random.sample(range(0, len(X)*len(X[1,:])), j*len(X));
    if j>0:
        for k in indices:
            if(randrange(0,1)==0):
            XNoise[np.unravel_index(k,(70000,784))]=255

    plt.imshow(XNoise[1,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig('Noised'+str(j)+'.png')
    bdt = AdaBoostClassifier(n_estimators=i)
    bdt.fit(XNoise[0:testlen, :], Y[0:testlen])

    ypred =  bdt.predict(XNoise[testlen:,:])
    scores.append(precision_score(Y[testlen:],ypred, average='micro'))
    print(scores[-1])

plt.plot(range(1,20), scores)
plt.savefig('ADANoise.png')


for i in range(10,15):
    bdt = AdaBoostClassifier(n_estimators=i)
    bdt.fit(X[0:testlen, :], Y[0:testlen])

    ypred =  bdt.predict(X[testlen:,:])
    scores.append(precision_score(Y[testlen:],ypred, average='micro'))
scores=list()
for i in range(1,20):
    bdt = GradientBoostingClassifier(n_estimators=1,learning_rate=1.0,max_depth=3,)
    bdt.fit(X[0:testlen, :], Y[0:testlen])
    ypred =  bdt.predict(X[testlen:,:])
    scores.append(precision_score(Y[testlen:],ypred, average='micro'))
    print(scores[-1])

plt.plot(range(1,20), scores)
plt.show()



plot_step = 0.02
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

plt.scatter(X[:,0],X[:,1], c=Y)
plot_colors = "br"
#plt.show()

