from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# CHALLENGE - create 3 more classifiers...

# 1 Decision Tree classifier
dectree = tree.DecisionTreeClassifier()

# 2 KNeighbors classifier

kneigh = KNeighborsClassifier(n_neighbors=3)

# 3 Gaussian Naive Bayes classifier
GaussNB = GaussianNB()

# 4 Support Vector Machine classifier
svc_=SVC()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
dectree = dectree.fit(X, Y)
kneigh = kneigh.fit(X,Y)
GaussNB = GaussNB.fit(X,Y)
svc_ = svc_.fit(X,Y)

#predict the outputs 
prediction1 = dectree.predict([[190,70,43]])
prediction2 = kneigh.predict([[190,70,43]])
prediction3 = GaussNB.predict([[190,70,43]])
prediction4 = svc_.predict([[190,70,43]])

# 1 Decision Tree classifier
print('The Decision Tree classifier output is: ', prediction1) 

# 2 KNeighbors classifier
print('The KNeighbors classifier output is: ', prediction2)

# 3 Gaussian Naive Bayes classifier
print('The Gaussian Naive Bayes classifier output is: ',prediction3)

# 4 Support Vector Machine classifier
print('The Support Vector Machine classifier output is: ',prediction4)
