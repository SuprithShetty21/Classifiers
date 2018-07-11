from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score as acc_
# CHALLENGE - create 3 more classifiers...

# 1 Decision Tree classifier
dectree = tree.DecisionTreeClassifier()

# 2 KNeighbors classifier

kneigh = KNeighborsClassifier()

# 3 Gaussian Naive Bayes classifier
GaussNB = GaussianNB()

# 4 Supprot Vector Machine classifier 
svc=SVC()

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
svc = svc.fit(X,Y)

#predict the outputs 
prediction1 = dectree.predict(X)
prediction2 = kneigh.predict(X)
prediction3 = GaussNB.predict(X)
prediction4 = svc.predict(X)

#calculating the accuracies of each models
acc_dectree = acc_(Y,prediction1)*100
acc_kneigh = acc_(Y,prediction2)*100
acc_gauss = acc_(Y,prediction3)*100
acc_svc = acc_(Y,prediction4)*100

# CHALLENGE compare their results and print the best one!

#find the best performing model
index = np.argmax([acc_dectree,acc_kneigh,acc_gauss,acc_svc])

acc_dict = { 'DecisionTreeClassifier' : acc_dectree, 'KNeighborClassifier' : acc_kneigh, \
            'GaussianNaiveBayesClassifier' : acc_gauss, 'SupportVectorMachineClassifier' : acc_svc }

pred_dict = { 0 : 'DecisionTreeClassifier', 1 : 'KNeighborClassifier', \
            2 : 'GaussianNaiveBayesClassifier', 3 : 'SupportVectorMachineClassifier'}

print('The accuracies of the 4 models are : ')
for key in acc_dict:
    print(key+" ==> "+str(acc_dict[key]))

print('The best classifier is : {}'.format(pred_dict[index]))

