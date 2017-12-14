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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 1000)[:, np.newaxis]
Y = np.sin(X).ravel() + np.sin(7 * X).ravel() + rng.normal(0, 0.4, X.shape[0])

adaregr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                                  n_estimators=300, random_state=rng)

gregr = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=.01,
                                min_samples_split=5, loss='ls')

for t in range(00,300):
    Y[random.randrange(0,1000)] = Y[random.randrange(0,1000)]*random.randrange(-1,2,2)*1.5
    adaregr.fit(X, Y)
    ada1 = adaregr.predict(X)

    gregr.fit(X, Y)
    g1 = gregr.predict(X)

    plt.figure()
    plt.scatter(X, Y, c=(0.75,0.75,0.75), label="training samples")
    plt.plot(X, ada1, c="g", label="ADABoost", linewidth=2)
    plt.plot(X, g1, c="r", label="Gradient", linewidth=2)
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('ADABoost vs Gradient Boost Regression')
    plt.legend(loc=2)
    plt.savefig('frame'+str(t).zfill(3)+'.png')
    plt.clf()



for j in range(00,1000,100):
    score = 0
    i = 20;
    YNoise = Y;
    samples = random.sample(range(0, len(Y)),j);
    for k in samples:
        YNoise[k]=random.randrange(0,10);

    bdt = AdaBoostClassifier(n_estimators=i)
    bdt.fit(X[0:testlen, :], YNoise[0:testlen])

    ypred =  bdt.predict(X[testlen:,:])
    scores.append(precision_score(YNoise[testlen:],ypred, average='micro'))
    print("ADA:" + str(scores[-1]))
    
    bdt = GradientBoostingClassifier(n_estimators=5,learning_rate=1.0,max_depth=3,)
    bdt.fit(X[0:testlen, :], YNoise[0:testlen])
    ypred =  bdt.predict(X[testlen:,:])
    gscores.append(precision_score(YNoise[testlen:],ypred, average='micro'))
    print("GRA:" + str(gscores[-1]))

plt.plot(range(0,len(scores)), scores)
plt.savefig('GRANoise.png')

scores=list()
gscores=list()
for j in range(0,110,10):
    score = 0
    i = 20;
    XNoise = X;

    if j>0:
        XNoise[np.unravel_index(random.sample(range(0, len(X)*len(X[1,:])), round(j*len(X)/2)),(70000,784))]=0;
        XNoise[np.unravel_index(random.sample(range(0, len(X)*len(X[1,:])), round(j*len(X)/2)),(70000,784))]=255;

    #plt.imshow(XNoise[1,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
    #plt.axis('off')
    #plt.savefig('Noised'+str(j)+'.png')
    bdt = AdaBoostClassifier(n_estimators=i)
    bdt.fit(XNoise[0:testlen, :], Y[0:testlen])

    ypred =  bdt.predict(XNoise[testlen:,:])
    scores.append(precision_score(Y[testlen:],ypred, average='micro'))
    print("ADA:" + str(scores[-1]))
    
    bdt = GradientBoostingClassifier(n_estimators=5,learning_rate=1.0,max_depth=3,)
    bdt.fit(X[0:testlen, :], Y[0:testlen])
    ypred =  bdt.predict(X[testlen:,:])
    gscores.append(precision_score(Y[testlen:],ypred, average='micro'))
    print("GRA:" + str(gscores[-1]))

plt.plot(range(0,len(scores)), scores)
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

