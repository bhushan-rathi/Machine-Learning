
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

#[height, weight, shoesize]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#[Target]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#Training the classifiers 
clf = tree.DecisionTreeClassifier()
gnb = GaussianNB()
neigh  = NearestNeighbors()
mlpclf = MLPClassifier()  

#Recording Predictions
prediction1 = clf.fit(X, Y).predict([[200, 80, 45]])
prediction2 = gnb.fit(X, Y).predict([[200, 80, 45]])
prediction3 = neigh.fit(X).kneighbors([[200, 80, 45]], 1, return_distance=False )
prediction4 = mlpclf.fit(X, Y).predict([[200, 80, 45]])


print(prediction1)
print(prediction2)
print([Y[prediction3]])
print(prediction4)