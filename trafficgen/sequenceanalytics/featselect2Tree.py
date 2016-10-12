import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.system("rm ./entropy")
#create the training & test sets, skipping the header row with [1:]
featurename = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype=None)[0]
dataset = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype='f8')[1:]
dataset2 = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype=None)[1:]
#print dataset 
target = [x[len(x)-1] for x in dataset2]
train = [x[0:len(x)-2] for x in dataset]
print len(train[1])

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=5,
                              random_state=0)

forest.fit(train,target)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]



#print indices
# Print the feature ranking
#print("Feature ranking:")
print len(train)
#for f in range(len(train[1])):
#    print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],featurename[indices[f]]))


def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")


for f in range(len(train[1])):
      keys =  str(featurename[indices[f]]).split("_")
      appendToFile("entropy"," ".join(keys)+ " #SUP: "+ str(importances[indices[f]]))
      



model = SelectFromModel(forest, prefit=True)
X_new = model.transform(train)
print X_new.shape
clf = svm.SVC()
#clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, train, target, cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
appendToFile(sys.argv[1],""+str(scores.mean())+" "+str(scores))
#clf = RandomForestClassifier(n_estimators=100)
clf = svm.SVC(gamma=0.0000019073486328125,kernel='rbf',C=131072.0)
scores = cross_val_score(clf, X_new, target, cv=5)

print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
appendToFile(sys.argv[1],""+str(scores.mean())+" " + str(scores))



X_train, X_test, y_train, y_test = train_test_split(X_new, target, test_size=0.40, random_state=42)
clf = svm.SVC(gamma=0.0000019073486328125,kernel='rbf',C=131072.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
newscore = accuracy_score(y_test, y_pred)
print newscore
appendToFile(sys.argv[1],"Calc score: "+str(newscore))






