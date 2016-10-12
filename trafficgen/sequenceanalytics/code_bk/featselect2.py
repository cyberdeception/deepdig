
import numpy as np

from numpy import genfromtxt, savetxt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


#create the training & test sets, skipping the header row with [1:]
featurename = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype=None)[0]
dataset = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype='f8')[1:]
dataset2 = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype=None)[1:]
#print dataset 
target = [x[len(x)-1] for x in dataset2]
train = [x[0:len(x)-2] for x in dataset]
print len(train[1])

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(train,target)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print indices
# Print the feature ranking
print("Feature ranking:")
print len(train)
for f in range(len(train[1])):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],featurename[indices[f]]))


def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")


for f in range(len(train[1])):
      keys =  str(featurename[indices[f]]).split("_")
      appendToFile("entropy"," ".join(keys)+ " #SUP: "+ str(importances[indices[f]]))
      

