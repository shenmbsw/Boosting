import scipy.io
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.utils import shuffle

districts = list()
X=np.empty([0,2])
Y=list()
data = scipy.io.loadmat(sys.argv[1])
for i in range(len(data['latitude'])):
    if data['neighbourhood'][i][0][0] not in districts:
        districts.append(data['neighbourhood'][i][0][0])

    X = np.row_stack((X,[data['latitude'][i][0],data['longitude'][i][0]]))
    Y.append(districts.index(data['neighbourhood'][i][0][0]))


X,Y = shuffle(X,Y,random_state=0)
testlen=round(0.7*len(Y))
scores = list()
for i in range(190,200):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=i,algorithm="SAMME")
    bdt.fit(X[0:testlen, :], Y[0:testlen])

    ypred =  bdt.predict(X[testlen:,:])
    scores.append(precision_score(Y[testlen:],ypred, average='micro'))
    print(scores[-1])


#plt.plot(range(1,200), scores)
#plt.show()


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),n_estimators=500)
bdt.fit(X[0:testlen, :], Y[0:testlen])
print(precision_score(Y[testlen:],ypred, average='micro'))

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
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.title('Boston Neighborhoods')
plt.savefig('boston.png')

plt.show()
