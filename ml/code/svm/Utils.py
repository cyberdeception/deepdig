#__author__ = 'khaled'
import os

from sklearn.decomposition import PCA
import config

#import Datastore

#import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn import preprocessing
from sklearn import linear_model

#import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV, LogisticRegression

import classifiers.wekaAPI
import itertools
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay

import random

from sklearn.metrics import roc_curve, auc

from scipy.stats import gaussian_kde

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class Utils:
    @staticmethod
    def readFile(fileName):
        fileLines = [line.strip() for line in open(fileName)]
        fileList = []
        for fileLine in fileLines:
            fileList.append(fileLine)

        return fileList

    # order instances by class label
    @staticmethod
    def readFile1(fileName):
        fileLines = [line.strip() for line in open(fileName)]
        fileList = []
        instancesList = []

        for fileLine in fileLines:
            if fileLine.startswith("@"):
                fileList.append(fileLine)
            else:
                lineList = fileLine.split(",")
                instancesList.append(lineList) # list of lists

        sortedInstancesList = sorted(instancesList, key=lambda x: x[lineList.__len__() - 1]) # sort by webpage label

        for instance in sortedInstancesList:
            fileList.append(",".join(instance))

        return fileList


    # Combine training and testing files and order instances by class label
    @staticmethod
    def readFile2(trainFileName, testFileName):
        trainFileLines = [line.strip() for line in open(trainFileName)]
        testFileLines = [line.strip() for line in open(testFileName)]
        fileList = []
        instancesList = []

        for fileLine in trainFileLines:
            if fileLine.startswith("@"):
                fileList.append(fileLine)
            else:
                lineList = fileLine.split(",")
                instancesList.append(lineList) # list of lists

        for fileLine in testFileLines:
            # Take instances from the testing file
            if not (fileLine.startswith("@") or fileLine.startswith("Direction")):
                lineList = fileLine.split(",")
                instancesList.append(lineList) # list of lists

        sortedInstancesList = sorted(instancesList, key=lambda x: x[lineList.__len__() - 1]) # sort by webpage label

        for instance in sortedInstancesList:
            fileList.append(",".join(instance))

        return fileList


    # Combine training and testing files and order instances by class label
    # Getting #instancesPerClass only from training file
    @staticmethod
    def readFile3(trainFileName, testFileName, instancesPerClass):
        trainFileLines = [line.strip() for line in open(trainFileName)]
        testFileLines = [line.strip() for line in open(testFileName)]
        fileList = []
        instancesList = []

        # for training file
        entityCtr = 0
        candidateList = []

        # get curr website and curr entity
        for line in trainFileLines:
            if not line[0] == '@':
                lineArray = line.split(",")
                currentWebsite = lineArray[-1]
                currentEntity = lineArray[1]
                break

        lessInstancesWebsites = []

        for line in trainFileLines:
            if line[0] == '@':
                fileList.append(line)
            else:
                #print line
                lineArray = line.split(",")
                website = lineArray[-1]
                entity = lineArray[1]
                if entity == currentEntity:
                    if entityCtr < instancesPerClass:
                        candidateList.append(lineArray)
                else:
                    entityCtr = entityCtr + 1

                    # for the first line in next entity
                    if entityCtr < instancesPerClass:
                        candidateList.append(lineArray)

                    currentEntity = entity
                    if website != currentWebsite:
                        if entityCtr >= instancesPerClass:
                            for instance in candidateList:
                                instancesList.append(instance)
                        #if entityCtr == 7:
                            #print currentWebsite + " " + str(classes.index(currentWebsite))
                        entityCtr = 0
                        currentWebsite = website
                        candidateList = []


        if entityCtr >= instancesPerClass:
            for instance in candidateList:
                instancesList.append(instance)

        # same for testing file
        entityCtr = 0
        candidateList = []
        testingInstancesList = [] # as websites are not ordered in CPD testing files

        for fileLine in testFileLines:
            # Take instances from the testing file
            if not (fileLine.startswith("@") or fileLine.startswith("Direction")):
                lineList = fileLine.split(",")
                testingInstancesList.append(lineList) # list of lists

        sortedTestingInstancesList = sorted(testingInstancesList, key=lambda x: x[lineList.__len__() - 1]) # sort by webpage label

        # get curr website and curr entity
        for line in sortedTestingInstancesList:
            line = ",".join(line)
            if not (line.startswith("@") or line.startswith("Direction")):
                lineArray = line.split(",")
                currentWebsite = lineArray[-1]
                currentEntity = lineArray[1]
                break


        for line in sortedTestingInstancesList:
            line = ",".join(line)
            if not (line.startswith("@") or line.startswith("Direction")):
                #print line
                lineArray = line.split(",")
                website = lineArray[-1]
                entity = lineArray[1]
                if entity == currentEntity:
                    if entityCtr < instancesPerClass:
                        candidateList.append(lineArray)
                else:
                    entityCtr = entityCtr + 1

                    # for the first line in next entity
                    if entityCtr < instancesPerClass:
                        candidateList.append(lineArray)

                    currentEntity = entity
                    if website != currentWebsite:
                        if entityCtr >= instancesPerClass:
                            for instance in candidateList:
                                instancesList.append(instance)
                        #if entityCtr == 7:
                            #print currentWebsite + " " + str(classes.index(currentWebsite))
                        entityCtr = 0
                        currentWebsite = website
                        candidateList = []

        if entityCtr >= instancesPerClass:
            for instance in candidateList:
                instancesList.append(instance)

        '''
        for fileLine in trainFileLines:
            if fileLine.startswith("@"):
                fileList.append(fileLine)
            else:
                lineList = fileLine.split(",")
                instancesList.append(lineList) # list of lists

        for fileLine in testFileLines:
            # Take instances from the testing file
            if not (fileLine.startswith("@") or fileLine.startswith("Direction")):
                lineList = fileLine.split(",")
                instancesList.append(lineList) # list of lists
        '''

        #print instancesList

        sortedInstancesList = sorted(instancesList, key=lambda x: x[lineArray.__len__() - 1]) # sort by webpage label

        # Return list that has equal number of training and testing instances for each class (instancesPerClass * 2)
        entityCtr = 0
        candidateList = []

        sortedInstancesListFull = []

        # get curr website and curr entity
        for line in sortedInstancesList:
            line = ",".join(line)
            if not (line.startswith("@") or line.startswith("Direction")):
                lineArray = line.split(",")
                currentWebsite = lineArray[-1]
                currentEntity = lineArray[1]
                break

        #print sortedInstancesList

        for line in sortedInstancesList:
            line = ",".join(line)
            if not (line.startswith("@") or line.startswith("Direction")):
                #print line
                lineArray = line.split(",")
                website = lineArray[-1]
                entity = lineArray[1]
                if entity == currentEntity:
                    if entityCtr < (instancesPerClass * 2):
                        candidateList.append(lineArray)
                else:
                    entityCtr = entityCtr + 1
                    # for the first line in next entity
                    if entityCtr < (instancesPerClass * 2):
                        candidateList.append(lineArray)

                    currentEntity = entity
                    if website != currentWebsite:
                        if entityCtr >= (instancesPerClass * 2):
                            for instance in candidateList:
                                sortedInstancesListFull.append(instance)
                        #if entityCtr == 7:
                            #print currentWebsite + " " + str(classes.index(currentWebsite))
                        entityCtr = 0
                        currentWebsite = website
                        candidateList = []

        if entityCtr >= (instancesPerClass * 2):
            for instance in candidateList:
                sortedInstancesListFull.append(instance)

        #print sortedInstancesListFull
        for instance in sortedInstancesListFull:
            fileList.append(",".join(instance))

        return fileList


    '''
    @staticmethod
    def calcPCA():
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        pca = PCA(n_components=2)
        sklearn_transf = pca.fit(X)
        print sklearn_transf.components_
        print sklearn_transf.components_[0,0]
        print(pca.explained_variance_ratio_)
    '''

    @staticmethod
    def calcPCA(files):
        '''
        #X = np.array([[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]])
        #instancesList = [[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]]
        print files
        instancesList = []
        instancesList.append([-1, -1, 1])
        instancesList.append([-2, -1, 5])
        instancesList.append([-3, -2, 6])
        instancesList.append([1, 1, 7])
        instancesList.append([2, 1, 8])
        instancesList.append([3, 2, 5])
        print instancesList
        X = np.array(instancesList)
        pca = PCA(n_components=2)
        sklearn_transf = pca.fit(X)
        print sklearn_transf.components_
        print sklearn_transf.components_[0,0]
        print ""
        print "X",  X[0].T
        print "EigVec", sklearn_transf.components_[0]
        print X[0].T.dot(sklearn_transf.components_[0])
        #print(pca.explained_variance_ratio_)
        '''
        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])

        #print instancesList

        X = np.array(instancesList)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        #config.n_components_PCA = 39

        pca = PCA(n_components=config.n_components_PCA)
        sklearn_transf = pca.fit(X)
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        #print pca.explained_variance_ratio_

        print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_PCA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        for line in trainList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_PCA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(sklearn_transf.components_[j])) # dot product for each instance and the eigen vector associated with highest eigen values

                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)


        for line in testList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_PCA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(sklearn_transf.components_[j]))

                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcPCA2(files):

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        #print instancesList

        X = np.array(instancesList)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        #config.n_components_PCA = 39

        pca = PCA(n_components=config.n_components_PCA)
        sklearn_transf = pca.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])

        X_transformed = sklearn_transf.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        #print X_transformed[0:1,0:5]
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        print pca.explained_variance_ratio_

        print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_PCA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

        #for item in X_lda_sklearn:
        for i in xrange(len(X)):
            webpageInstance=[]
            for j in range(0,config.n_components_PCA):
                webpageInstance.append(X[i].T.dot(sklearn_transf.components_[j]))
            #print webpageInstance
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])

            ctr = ctr + 1


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcPCA3(files):

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        #print instancesList

        X = np.array(instancesList)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        #config.n_components_PCA = 39

        pca = PCA(n_components=config.n_components_PCA)
        sklearn_transf = pca.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])

        X_transformed = sklearn_transf.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        #print X_transformed[0:1,0:5]
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        print pca.explained_variance_ratio_

        print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_PCA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

        #for item in X_lda_sklearn:
        for i in xrange(len(X)):
            webpageInstance=[]
            for j in range(0,config.n_components_PCA):
                webpageInstance.append(X[i].T.dot(sklearn_transf.components_[j]))
            #print webpageInstance
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])

            ctr = ctr + 1


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcLDA(files): # Linear (Multi Class) Discriminant Analysis

        '''
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        #y = np.array([1, 1, 1, 2, 2, 2])
        y = np.array(['aa', 'aa', 'aa', 'bc', 'bc', 'bc'])
        clf = LDA()
        print clf.fit(X, y)
        print(clf.predict([[-0.8, -1]]))

        feature_dict = {i:label for i,label in zip(
            range(4),
              ('sepal length in cm',
              'sepal width in cm',
              'petal length in cm',
              'petal width in cm', ))}



        df = pd.io.parsers.read_csv(
                filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                header=None,
                sep=',',
                )
        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
        df.dropna(how="all", inplace=True) # to drop the empty line at file-end

        #print df.tail()




        X = df[[0,1,2,3]].values
        y = df['class label'].values

        #print y
        #print X



        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1

        #print y

        label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}


        sklearn_lda = LDA(n_components=2)
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        print X_lda_sklearn
        #print y
        Utils.plot_scikit_lda(X_lda_sklearn, y, label_dict, 'LDA - scikit-learn 15.2')



        #X = np.array([[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]])
        #instancesList = [[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]]
        print files
        instancesList = []
        instancesList.append([-1, -1, 1])
        instancesList.append([-2, -1, 5])
        instancesList.append([-3, -2, 6])
        instancesList.append([1, 1, 7])
        instancesList.append([2, 1, 8])
        instancesList.append([3, 2, 5])
        print instancesList
        X = np.array(instancesList)
        pca = PCA(n_components=2)
        sklearn_transf = pca.fit(X)
        print sklearn_transf.components_
        print sklearn_transf.components_[0,0]
        print ""
        print "X",  X[0].T
        print "EigVec", sklearn_transf.components_[0]
        print X[0].T.dot(sklearn_transf.components_[0])
        #print(pca.explained_variance_ratio_)
        '''
        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])

        #print instancesList

        X = np.array(instancesList)
        y = np.array(y)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        print X_lda_sklearn
        print y

        #config.n_components_PCA = 39

        #pca = PCA(n_components=config.n_components_PCA)
        #sklearn_transf = pca.fit(X)
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        #print pca.explained_variance_ratio_

        #print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

        #for item in X_lda_sklearn:
        for i in xrange(len(X_lda_sklearn)):
            #print  X_lda_sklearn[i][0]
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            ctr = ctr + 1


        '''
        for line in trainList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_PCA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(sklearn_transf.components_[j])) # dot product for each instance and the eigen vector associated with highest eigen values

                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)


        for line in testList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_PCA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(sklearn_transf.components_[j]))

                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)
        '''

        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]
        #'''

    @staticmethod
    def calcLDA2(files): # Linear (Multi Class) Discriminant Analysis

        '''
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        #y = np.array([1, 1, 1, 2, 2, 2])
        y = np.array(['aa', 'aa', 'aa', 'bc', 'bc', 'bc'])
        clf = LDA()
        print clf.fit(X, y)
        print(clf.predict([[-0.8, -1]]))

        feature_dict = {i:label for i,label in zip(
            range(4),
              ('sepal length in cm',
              'sepal width in cm',
              'petal length in cm',
              'petal width in cm', ))}



        df = pd.io.parsers.read_csv(
                filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                header=None,
                sep=',',
                )
        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
        df.dropna(how="all", inplace=True) # to drop the empty line at file-end

        #print df.tail()




        X = df[[0,1,2,3]].values
        y = df['class label'].values

        #print y
        #print X



        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1

        #print y

        label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}


        sklearn_lda = LDA(n_components=2)
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        print X_lda_sklearn
        #print y
        Utils.plot_scikit_lda(X_lda_sklearn, y, label_dict, 'LDA - scikit-learn 15.2')



        #X = np.array([[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]])
        #instancesList = [[-1, -1, 1], [-2, -1, 5], [-3, -2, 6], [1, 1, 7], [2, 1, 8], [3, 2, 5]]
        print files
        instancesList = []
        instancesList.append([-1, -1, 1])
        instancesList.append([-2, -1, 5])
        instancesList.append([-3, -2, 6])
        instancesList.append([1, 1, 7])
        instancesList.append([2, 1, 8])
        instancesList.append([3, 2, 5])
        print instancesList
        X = np.array(instancesList)
        pca = PCA(n_components=2)
        sklearn_transf = pca.fit(X)
        print sklearn_transf.components_
        print sklearn_transf.components_[0,0]
        print ""
        print "X",  X[0].T
        print "EigVec", sklearn_transf.components_[0]
        print X[0].T.dot(sklearn_transf.components_[0])
        #print(pca.explained_variance_ratio_)
        '''
        '''
        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])
        '''
        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        trainInstancesList = []
        testInstancesList = []
        classes = ""
        yTrain = []
        yTest = []

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTrain.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                testInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTest.append(line.split(",")[-1])

        #print instancesList

        X = np.array(trainInstancesList)
        y = np.array(yTrain)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        #print X_lda_sklearn
        #print y
        lda_sklearn = sklearn_lda.fit(X, y)
        print lda_sklearn.coef_[0]
        #print "score "  + str(sklearn_lda.score(testInstancesList,yTest))
        #print "predict "  + str(sklearn_lda.predict(testInstancesList))
        #print testInstancesList
        #print yTest
        #config.n_components_PCA = 39

        #pca = PCA(n_components=config.n_components_PCA)
        #sklearn_transf = pca.fit(X)
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        #print pca.explained_variance_ratio_

        #print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

        '''
        for i in xrange(len(X_lda_sklearn)):
            #print  X_lda_sklearn[i][0]
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            ctr = ctr + 1


        '''
        for line in trainList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_LDA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(lda_sklearn.coef_[j])) # dot product for each instance and the eigen vector associated with highest eigen values

                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)


        '''
        for line in testList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_LDA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(lda_sklearn.coef_[j]))

                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)
        '''

        testX = np.array(testInstancesList)

        preprocessing.scale(testX, axis=0, with_mean=True, with_std=True, copy=False)


        for i in xrange(len(testX)):
            webpageInstance=[]
            for j in range(0,config.n_components_LDA):
                webpageInstance.append(testX[i].T.dot(lda_sklearn.coef_[j]))

            newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + yTest[i])

        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]



    @staticmethod
    def calcLDA3(files): # Linear (Multi Class) Discriminant Analysis

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        labels = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        X = np.array(instancesList)
        y = np.array(labels)

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #print X[0:1,0:1]
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X[0:1,0:1]
        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        #X_lda_sklearn = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        lda_sklearn = sklearn_lda.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        transformedX = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        print transformedX

        #print lda_sklearn.coef_[0] # first eigen vector
        ##print y

        #config.n_components_PCA = 39

        #pca = PCA(n_components=config.n_components_PCA)
        #sklearn_transf = pca.fit(X)
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_
        #print pca.explained_variance_ratio_

        #print ('explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0


        #for item in X_lda_sklearn:
        for i in xrange(len(X)):
            webpageInstance=[]
            for j in range(0,config.n_components_LDA):
                webpageInstance.append(X[i].T.dot(lda_sklearn.coef_[j]))
            print webpageInstance
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])

            ctr = ctr + 1

        '''
        for i in xrange(len(X_lda_sklearn)):
            #print  X_lda_sklearn[i][0]
            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
                newTrainList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            else:
                newTestList.append(','.join([str(k) for k in X_lda_sklearn[i]]) + ',' + y[i])
            ctr = ctr + 1


        for line in trainList:
            if line[0] != '@':
                webpage=line.split(",")[-1]

                for j in range(0,config.n_components_LDA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(lda_sklearn.coef_[j])) # dot product for each instance and the eigen vector associated with highest eigen values

                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)



        for line in testList:
            if line[0] != '@':
                webpage=line.split(",")[-1]
                webpageInstance=[]
                for j in range(0,config.n_components_LDA):
                    webpageInstance.append(np.array([float(i) for i in line.split(",")[:-1]]).T.dot(lda_sklearn.coef_[j]))

                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + webpage)
        '''


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcLDA4(files): # Linear (Multi Class) Discriminant Analysis

        #trainList = Utils.readFile1(files[0])
        #testList = Utils.readFile1(files[1])
        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        instancesList = []
        classes = ""
        labels = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        X = np.array(instancesList)
        y = np.array(labels)

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #print X[0:1,0:1]
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        #X_lda_sklearn = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        lda_sklearn = sklearn_lda.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])


        #transformedX = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        #print transformedX

        X_train_lda_sklearn = lda_sklearn.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        X_test_lda_sklearn = lda_sklearn.transform(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:])

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        #print X_train_lda_sklearn

        for i in xrange(len(X_train_lda_sklearn)):
            #print X_train_lda_sklearn[i]
            newTrainList.append(','.join([str(k) for k in X_train_lda_sklearn[i]]) + ',' + y[i])

        for i in xrange(len(X_test_lda_sklearn)):
            newTestList.append(','.join([str(k) for k in X_test_lda_sklearn[i]]) + ',' + y[i+len(X_train_lda_sklearn)])

        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

#    @staticmethod
#    def calcLDA5(files): # Linear (Multi Class) Discriminant Analysis
#
#        feature_dict = {i:label for i,label in zip(
#            range(4),
#              ('sepal length in cm',
#              'sepal width in cm',
#              'petal length in cm',
#              'petal width in cm', ))}
#
#
#
#        df = pd.io.parsers.read_csv(
#                filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
#                header=None,
#                sep=',',
#                )
#        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
#        df.dropna(how="all", inplace=True) # to drop the empty line at file-end
#
#        #print df.tail()
#
#
#
#
#        X = df[[0,1,2,3]].values
#        y = df['class label'].values
#
#        print y
#        print X
#
#
#        enc = LabelEncoder()
#        label_encoder = enc.fit(y)
#        y = label_encoder.transform(y) + 1
#
#        print y
#
#        label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
#
#        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
#
#        np.set_printoptions(precision=4)
#
#        mean_vectors = [] # For each class
#        for cl in range(1,4): # three classes
#            mean_vectors.append(np.mean(X[y==cl], axis=0))
#            print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
#
#        # Within-class scatter matrix Sw
#        S_W = np.zeros((4,4)) # square matrix of number of features
#        for cl,mv in zip(range(1,4), mean_vectors):
#            class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
#            for row in X[y == cl]:
#                row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
#                class_sc_mat += (row-mv).dot((row-mv).T)
#            S_W += class_sc_mat                             # sum class scatter matrices
#        print('within-class Scatter Matrix:\n', S_W)
#
#        # Between-class scatter matrix Sb
#        overall_mean = np.mean(X, axis=0)
#
#        S_B = np.zeros((4,4)) # square matrix of number of features
#        for i,mean_vec in enumerate(mean_vectors):
#            n = X[y==i+1,:].shape[0] # number of instances for each class or group
#            mean_vec = mean_vec.reshape(4,1) # make column vector
#            overall_mean = overall_mean.reshape(4,1) # make column vector
#            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
#
#        print('between-class Scatter Matrix:\n', S_B)
#
#        #Solving the generalized eigenvalue problem for the matrix SW-1SB
#        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
#
#        for i in range(len(eig_vals)):
#            eigvec_sc = eig_vecs[:,i].reshape(4,1)
#            print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
#            print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
#
#        #Checking the eigenvector-eigenvalue calculation
#        for i in range(len(eig_vals)):
#            eigv = eig_vecs[:,i].reshape(4,1)
#            np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
#                                                 eig_vals[i] * eigv,
#                                                 decimal=6, err_msg='', verbose=True)
#        print('ok')
#
#
#        #Sorting the eigenvectors by decreasing eigenvalues
#        # Make a list of (eigenvalue, eigenvector) tuples
#        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#
#        # Sort the (eigenvalue, eigenvector) tuples from high to low
#        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
#
#        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
#
#        print('Eigenvalues in decreasing order:\n')
#        for i in eig_pairs:
#            print(i[0])
#
#
#        print('Variance explained:\n')
#        eigv_sum = sum(eig_vals)
#        for i,j in enumerate(eig_pairs):
#            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
#
#        W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
#        print('Matrix W:\n', W.real)
#
#        #Transforming the samples onto the new subspace
#        X_lda = X.dot(W)
#
#        print X_lda
#
#        '''
#
#
#
#        #trainList = Utils.readFile1(files[0])
#        #testList = Utils.readFile1(files[1])
#        trainList = Utils.readFile(files[0])
#        testList = Utils.readFile(files[1])
#
#        instancesList = []
#        classes = ""
#        labels = []
#        for line in trainList:
#            if line[0] == '@':
#                 if line.lower().startswith("@attribute class"):
#                     classes = line.split(" ")[2]
#            else:
#                #instancesList.append(float(line.split(",")[:-1]))
#                instancesList.append([float(i) for i in line.split(",")[:-1]])
#                labels.append(line.split(",")[-1])
#
#        for line in testList:
#            if line[0] != '@':
#                instancesList.append([float(i) for i in line.split(",")[:-1]])
#                labels.append(line.split(",")[-1])
#
#        X = np.array(instancesList)
#        y = np.array(labels)
#
#        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
#        #print X[0:1,0:1]
#        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
#        #print X
#        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
#        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
#        #X_lda_sklearn = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
#        lda_sklearn = sklearn_lda.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
#
#
#        #transformedX = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
#        #print transformedX
#
#        X_train_lda_sklearn = lda_sklearn.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
#        X_test_lda_sklearn = lda_sklearn.transform(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:])
#
#        newTrainList = []
#        newTestList = []
#
#        newTrainList.append('@RELATION sites')
#        newTestList.append('@RELATION sites')
#
#        for i in range(0,config.n_components_LDA):
#            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
#            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')
#
#        newTrainList.append('@ATTRIBUTE class '+classes)
#        newTestList.append('@ATTRIBUTE class '+classes)
#
#        newTrainList.append('@DATA')
#        newTestList.append('@DATA')
#
#        #print X_train_lda_sklearn
#
#        for i in xrange(len(X_train_lda_sklearn)):
#            #print X_train_lda_sklearn[i]
#            newTrainList.append(','.join([str(k) for k in X_train_lda_sklearn[i]]) + ',' + y[i])
#
#        for i in xrange(len(X_test_lda_sklearn)):
#            newTestList.append(','.join([str(k) for k in X_test_lda_sklearn[i]]) + ',' + y[i+len(X_train_lda_sklearn)])
#
#        # writing the new training file (with lower dimensions)
#        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
#        fnewTrain = open(fnewTrainName, 'w')
#        for item in newTrainList:
#            fnewTrain.write(item+'\n')
#
#        fnewTrain.close()
#
#        # writing the new testing file (with lower dimensions)
#        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
#        fnewTest = open(fnewTestName, 'w')
#        for item in newTestList:
#            fnewTest.write(item+'\n')
#
#        fnewTest.close()
#
#        return [fnewTrainName,fnewTestName]
#    '''
#
    @staticmethod
    def calcLDA6(files): # Linear (Multi Class) Discriminant Analysis
        '''
        feature_dict = {i:label for i,label in zip(
            range(4),
              ('sepal length in cm',
              'sepal width in cm',
              'petal length in cm',
              'petal width in cm', ))}



        df = pd.io.parsers.read_csv(
                filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                header=None,
                sep=',',
                )
        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
        df.dropna(how="all", inplace=True) # to drop the empty line at file-end

        #print df.tail()




        X = df[[0,1,2,3]].values
        y = df['class label'].values

        '''

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        labels = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])


        X = np.array(instancesList)
        y = np.array(labels)

        #print X
        #print y

        std_vectors = np.std(X, axis=0)

        print std_vectors



        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1

        #print y

        #label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

        #np.set_printoptions(precision=4)

        mean_vectors = [] # For each class
        for cl in range(1,config.BUCKET_SIZE+1): # number of classes
            mean_vectors.append(np.mean(X[y==cl], axis=0))
            print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))

        # Within-class scatter matrix Sw
        S_W = np.zeros((len(instancesList[0]),len(instancesList[0]))) # square matrix of number of features
        for cl,mv in zip(range(1,config.BUCKET_SIZE+1), mean_vectors):
            class_sc_mat = np.zeros((len(instancesList[0]),len(instancesList[0])))                  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(len(instancesList[0]),1), mv.reshape(len(instancesList[0]),1) # make column vectors
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W += class_sc_mat                             # sum class scatter matrices
        print('within-class Scatter Matrix:\n', S_W)

        # Between-class scatter matrix Sb
        overall_mean = np.mean(X, axis=0)

        S_B = np.zeros((len(instancesList[0]),len(instancesList[0]))) # square matrix of number of features
        for i,mean_vec in enumerate(mean_vectors):
            n = X[y==i+1,:].shape[0] # number of instances for each class or group
            mean_vec = mean_vec.reshape(len(instancesList[0]),1) # make column vector
            overall_mean = overall_mean.reshape(len(instancesList[0]),1) # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        print('between-class Scatter Matrix:\n', S_B)

        #print S_W

        #print np.linalg.inv(S_W) # problem here as there is a det that is zero

        #Solving the generalized eigenvalue problem for the matrix SW-1SB
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(len(instancesList[0]),1)
            print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
            print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

        #Checking the eigenvector-eigenvalue calculation
        for i in range(len(eig_vals)):
            eigv = eig_vecs[:,i].reshape(len(instancesList[0]),1)
            np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                                 eig_vals[i] * eigv,
                                                 decimal=6, err_msg='', verbose=True)
        print('ok')


        #Sorting the eigenvectors by decreasing eigenvalues
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues

        print('Eigenvalues in decreasing order:\n')
        for i in eig_pairs:
            print(i[0])


        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

        W = np.hstack((eig_pairs[0][1].reshape(len(instancesList[0]),1), eig_pairs[1][1].reshape(len(instancesList[0]),1)))
        print('Matrix W:\n', W.real)

        #Transforming the samples onto the new subspace
        X_lda = X.dot(W)

        print X_lda

        '''



        #trainList = Utils.readFile1(files[0])
        #testList = Utils.readFile1(files[1])
        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        instancesList = []
        classes = ""
        labels = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        X = np.array(instancesList)
        y = np.array(labels)

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #print X[0:1,0:1]
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        #X_lda_sklearn = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        lda_sklearn = sklearn_lda.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])


        #transformedX = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        #print transformedX

        X_train_lda_sklearn = lda_sklearn.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        X_test_lda_sklearn = lda_sklearn.transform(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:])

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        #print X_train_lda_sklearn

        for i in xrange(len(X_train_lda_sklearn)):
            #print X_train_lda_sklearn[i]
            newTrainList.append(','.join([str(k) for k in X_train_lda_sklearn[i]]) + ',' + y[i])

        for i in xrange(len(X_test_lda_sklearn)):
            newTestList.append(','.join([str(k) for k in X_test_lda_sklearn[i]]) + ',' + y[i+len(X_train_lda_sklearn)])

        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]
    '''

    @staticmethod
    def calcQDA(files): # Linear (Multi Class) Discriminant Analysis

        #trainList = Utils.readFile1(files[0])
        #testList = Utils.readFile1(files[1])
        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        instancesList = []
        classes = ""
        labels = []
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                labels.append(line.split(",")[-1])

        X = np.array(instancesList)
        y = np.array(labels)

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #print X[0:1,0:1]
        preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        #sklearn_lda = LDA(solver='svd',n_components=config.n_components_LDA)
        sklearn_lda = QDA()
        #sklearn_lda = LDA(solver='eigen',n_components=config.n_components_LDA)
        #X_lda_sklearn = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        lda_sklearn = sklearn_lda.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])


        #transformedX = sklearn_lda.fit_transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        #print transformedX

        X_train_lda_sklearn = lda_sklearn.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        X_test_lda_sklearn = lda_sklearn.transform(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:])

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_LDA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        #print X_train_lda_sklearn

        for i in xrange(len(X_train_lda_sklearn)):
            #print X_train_lda_sklearn[i]
            newTrainList.append(','.join([str(k) for k in X_train_lda_sklearn[i]]) + ',' + y[i])

        for i in xrange(len(X_test_lda_sklearn)):
            newTestList.append(','.join([str(k) for k in X_test_lda_sklearn[i]]) + ',' + y[i+len(X_train_lda_sklearn)])

        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_LDA_'+str(config.n_components_LDA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    '''
    @staticmethod
    def plot_scikit_lda(X, title, y, label_dict, mirror=1):

        ax = plt.subplot(111)
        for label,marker,color in zip(
            range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

            plt.scatter(x=X[:,0][y == label]*mirror,
                    y=X[:,1][y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

        plt.xlabel('LD1')
        plt.ylabel('LD2')

        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title(title)

        # hide axis ticks
        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.grid()
        plt.tight_layout
        plt.show()
    '''


    @staticmethod
    def calcLasso(files):

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])
                y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])
                y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        #print instancesList

        X = np.array(instancesList) #.astype(np.float)
        y = np.array(y).astype(np.float)

        clf = linear_model.Lasso(alpha = 0.1)
        clf.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])

        print(clf.coef_)
        print(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        print(clf.predict(X[config.BUCKET_SIZE * config.NUM_TRAINING_TRACES]))
        print(clf.predict(X[(config.BUCKET_SIZE * config.NUM_TRAINING_TRACES)+1]))
        print(clf.predict(X[(config.BUCKET_SIZE * config.NUM_TRAINING_TRACES)+2]))
        print(clf.predict(X[(config.BUCKET_SIZE * config.NUM_TRAINING_TRACES)+3]))
#        model = SelectFromModel(clf, prefit=True)
#        X_new = model.transform(X)
#        X_new.shape

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_PCA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classes)
        newTestList.append('@ATTRIBUTE class '+classes)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

#        #for item in X_lda_sklearn:
#        for i in xrange(len(X)):
#            webpageInstance=[]
#            for j in range(0,config.n_components_PCA):
#                webpageInstance.append(X[i].T.dot(sklearn_transf.components_[j]))
#            #print webpageInstance
#            if ctr < config.BUCKET_SIZE * config.NUM_TRAINING_TRACES:
#                newTrainList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])
#            else:
#                newTestList.append(','.join([str(k) for k in webpageInstance]) + ',' + y[i])
#
#            ctr = ctr + 1
#
#
#        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_lasso_.arff'
#        fnewTrain = open(fnewTrainName, 'w')
#        for item in newTrainList:
#            fnewTrain.write(item+'\n')
#
#        fnewTrain.close()
#
#        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_lasso.arff'
#        fnewTest = open(fnewTestName, 'w')
#        for item in newTestList:
#            fnewTest.write(item+'\n')
#
#        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcLasso2(files):
        # Author: Manoj Kumar <mks542@nyu.edu>
        # License: BSD 3 clause

        print(__doc__)



        # Load the boston dataset.
        boston = load_boston()
        X, y = boston['data'], boston['target']

        print 'X'
        print X
        print '\n'
        print 'y'
        print y

        X = X[0:10,0:5]
        y = y[0:10]

        print 'X'
        print X
        print '\n'
        print 'y'
        print y

        # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
        clf = LassoCV()

        # Set a minimum threshold of 0.25
        sfm = SelectFromModel(clf, threshold=0.25)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]

        # Reset the threshold till the number of features equals two.
        # Note that the attribute can be set directly instead of repeatedly
        # fitting the metatransformer.
        while n_features > 2:
            sfm.threshold += 0.1
            X_transform = sfm.transform(X)
            n_features = X_transform.shape[1]

        print 'X'
        print X
        print '\n'
        print 'y'
        print y


        print 'X_transform[:, 0]'
        print X_transform[:, 0]
        print '\n'
        print 'X_transform[:, 1]'
        print X_transform[:, 1]
        #print '\n'
        #print 'X_transform[:, 2]'
        #print X_transform[:, 2]

        print '\n'
        print 'sfm.get_support'
        print sfm.get_support(indices=True)

        # Plot the selected two features from X.
        plt.title(
            "Features selected from Boston using SelectFromModel with "
            "threshold %0.3f." % sfm.threshold)
        feature1 = X_transform[:, 0]
        feature2 = X_transform[:, 1]
        plt.plot(feature1, feature2, 'r.')
        plt.xlabel("Feature number 1")
        plt.ylabel("Feature number 2")
        plt.ylim([np.min(feature2), np.max(feature2)])
        plt.show()




#-----------------
    @staticmethod
    def calcLasso3(files):

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])
                y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        for line in testList:
            if line[0] != '@':
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])
                y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        #print instancesList

        X = np.array(instancesList) #.astype(np.float)
        y = np.array(y).astype(np.float)



        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y
#
        ##X = X[0:10,0:5]
        ##y = y[0:10]
#
        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y

        # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
        clf = LassoCV(max_iter=10000)
        #clf = LassoCV(max_iter=1000)

        # Set a minimum threshold of 0.25
        sfm = SelectFromModel(clf, threshold=0.25)
        #sfm = SelectFromModel(clf, threshold=0)

        sfm.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES], y[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        n_features = sfm.transform(X).shape[1]

        print 'n_features'
        print n_features
        print '\n'

        # Reset the threshold till the number of features equals two.
        # Note that the attribute can be set directly instead of repeatedly
        # fitting the metatransformer.
        #while n_features > 2:
        while n_features > config.lasso:
            sfm.threshold += 0.1
            X_transform = sfm.transform(X)
            n_features = X_transform.shape[1]

        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y

        #print '\n'
        #print 'sfm.get_support'
        #print sfm.get_support(indices=True)
#
        #print 'X_transform[:, 0]'
        #print X_transform[:, 0]
        #print '\n'
        #print 'X_transform[:, 1]'
        #print X_transform[:, 1]
        ##print '\n'
        ##print 'X_transform[:, 2]'
        ##print X_transform[:, 2]

        print '\n'
        print 'sfm.get_support'
        print sfm.get_support(indices=True)

        # Plot the selected two features from X.
        plt.title(
            "Features selected from Boston using SelectFromModel with "
            "threshold %0.3f." % sfm.threshold)
        feature1 = X_transform[:, 0]
        feature2 = X_transform[:, 1]
        plt.plot(feature1, feature2, 'r.')
        plt.xlabel("Feature number 1")
        plt.ylabel("Feature number 2")
        plt.ylim([np.min(feature2), np.max(feature2)])
        plt.show()


#-----------------
    @staticmethod
    def calcLogisticRegression(files):

        trainList = Utils.readFile1(files[0])
        #testList = Utils.readFile1(files[1])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])
                #y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        #for line in testList:
        #    if line[0] != '@':
        #        instancesList.append([float(i) for i in line.split(",")[:-1]])
        #        y.append(line.split(",")[-1])
        #        #y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

        #print instancesList

        X = np.array(instancesList) #.astype(np.float)
        y = np.array(y)

        print 'File'
        print files[0]

        print 'n_features original:'
        print len(X[0])
        print '\n'



        clf = LogisticRegression(penalty='l2',max_iter=1000,solver='newton-cg') # solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}



        # Set a minimum threshold of 0.25
        #sfm = SelectFromModel(clf, threshold=0.25)
        sfm = SelectFromModel(clf, threshold=config.lasso)
        #sfm = SelectFromModel(clf, threshold=0)

        sfm.fit(X, y) # take training data only
        n_features = sfm.transform(X).shape[1]

        #config.Num_Features_Selected = 100 # for testing

        if config.Num_Features_Selected != 0: # If we want a fixed number of features
            while n_features > config.Num_Features_Selected:
            #while n_features > config.lasso:
                sfm.threshold += 0.1
                X_transform = sfm.transform(X)
                n_features = X_transform.shape[1]


        print 'n_features selected:'
        print n_features
        print '\n'

        print '\n'
        print 'sfm.get_support'
        print sfm.get_support(indices=True)

        selectedFeaturesList = np.array(sfm.get_support(indices=True)).tolist()

        print selectedFeaturesList

        #'#All Features','Threshold','#Selected Features','Accuracy All','Accuracy Selected'
        output = [len(X[0])]
        output.append(config.lasso)
        output.append(n_features)

        Utils.__applyFeatureSelection(files[0], files[1], selectedFeaturesList, output)


#-----------------
    @staticmethod
    def  __applyFeatureSelection(trainingFilename, testingFilename, featuresList, output):


    #    if not os.path.exists(outputFoldername):
    #        os.mkdir(outputFoldername)
    #    else:
    #        shutil.rmtree(outputFoldername) # delete and remake folder
    #        os.mkdir(outputFoldername)




        [accuracy,debugInfo] = wekaAPI.execute( trainingFilename,
                                 testingFilename,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072'] ) # Cost
        #print outputFoldername
        print 'accuracy before feature selection ' + str(accuracy)

        output.append(str(accuracy))

        trainList = Utils.readFile(trainingFilename)
        testList = Utils.readFile(testingFilename)
        #featuresList = Utils.readFile(featuresFilename)
        #featuresList = featuresList[0].split("{")[1].split("}")[0].split(",")
        featuresList = [int(i) for i in featuresList]
        featuresList = sorted(featuresList)
        #print featuresList.__contains__(4)


        newTrainList = []
        newTestList = []

        for i in range(len(trainList)):
            if trainList[i].startswith('@'):
                if trainList[i].startswith('@ATTRIBUTE'):
                    if featuresList.__contains__(i-1): # featuresList index startr from 0, arff features(line) starts from 1
                        #print trainList[i].split(" ")[1]
                        newTrainList.append(trainList[i])
                else:
                     newTrainList.append(trainList[i])
                if trainList[i].startswith('@ATTRIBUTE class'):
                    newTrainList.append(trainList[i])
            else:
                newInstance = []
                instanceSplit = trainList[i].split(",")
                newInstance = [instanceSplit[j] for j in featuresList] # take indecies from featuresList whose index starts from 0
                #newInstance = [instanceSplit[j-1] for j in featuresList] # take indecies from featuresList whose index starts from 1
                #for j in range(len(instanceSplit)):
                #    if featuresList.__contains__(j+1):
                #        newInstance.append(instanceSplit[j])

                newInstance.append(instanceSplit[-1])
                newTrainList.append(",".join(newInstance))

        for i in range(len(testList)):
            if testList[i].startswith('@'):
                if testList[i].startswith('@ATTRIBUTE'):
                    if featuresList.__contains__(i-1): # featuresList index startr from 0, arff features(line) starts from 1
                        #print testList[i].split(" ")[1]
                        newTestList.append(testList[i])
                else:
                     newTestList.append(testList[i])
                if testList[i].startswith('@ATTRIBUTE class'):
                    newTestList.append(testList[i])
            else:
                newInstance = []
                instanceSplit = testList[i].split(",")

                newInstance = [instanceSplit[j] for j in featuresList] # take indecies from featuresList whose index starts from 0
                #newInstance = [instanceSplit[j-1] for j in featuresList] # take indecies from featuresList whose index starts from 1

                #for j in range(len(instanceSplit)):
                #    if featuresList.__contains__(j+1):
                #        newInstance.append(instanceSplit[j])

                newInstance.append(instanceSplit[-1])
                newTestList.append(",".join(newInstance))


        fnewTrainName = trainingFilename[:-5]+'_Features'+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = testingFilename[:-5]+'_Features'+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        [accuracy,debugInfo] = wekaAPI.execute( fnewTrainName,
                                 fnewTestName,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072'] ) # Cost
        print 'accuracy after feature selection ' + str(accuracy)

        output.append(str(accuracy))

        summary = ', '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileName(trainingFilename)

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''


#-----------------
    @staticmethod
    def calcLogisticRegressionTest(files):

        boston = load_boston()
        X, y = boston['data'], boston['target']



        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y
#
        X = X[0:20,0:5]
        y = y[0:20]
        y = np.array(y).astype(np.int)
#
        print 'X'
        print X
        print '\n'
        print 'y'
        print y

        # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
        clf = LogisticRegression(penalty='l2',max_iter=1000,solver='newton-cg') # solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}
        #clf = LassoCV(max_iter=1000)

        # Set a minimum threshold of 0.25
        sfm = SelectFromModel(clf, threshold=0.25)
        #sfm = SelectFromModel(clf, threshold=0)

        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]


        print 'n_features'
        print n_features
        print '\n'

        # Reset the threshold till the number of features equals two.
        # Note that the attribute can be set directly instead of repeatedly
        # fitting the metatransformer.
        while n_features > 2:
        #while n_features > config.lasso:
            sfm.threshold += 0.1
            X_transform = sfm.transform(X)
            n_features = X_transform.shape[1]

        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y

        #print '\n'
        #print 'sfm.get_support'
        #print sfm.get_support(indices=True)
#
        print 'X_transform[:, 0]'
        print X_transform[:, 0]
        print '\n'
        print 'X_transform[:, 1]'
        print X_transform[:, 1]
        print '\n'
        ##print 'X_transform[:, 2]'
        ##print X_transform[:, 2]

        print '\n'
        print 'sfm.get_support'
        print sfm.get_support(indices=True)

        # Plot the selected two features from X.
        plt.title(
            "Features selected from Boston using SelectFromModel with "
            "threshold %0.3f." % sfm.threshold)
        feature1 = X_transform[:, 0]
        feature2 = X_transform[:, 1]
        plt.plot(feature1, feature2, 'r.')
        plt.xlabel("Feature number 1")
        plt.ylabel("Feature number 2")
        plt.ylim([np.min(feature2), np.max(feature2)])
        plt.show()


    @staticmethod
    def getOutputFileName(arffFileName):
        #outputFilenameArray = ['resultsRegression',
        #                       'k'+str(config.BUCKET_SIZE),
        #                       'c'+str(config.COUNTERMEASURE),
        #                       'd'+str(config.DATA_SOURCE),
        #                       'C'+str(config.CLASSIFIER),
        #                       'N'+str(config.TOP_N),
        #                       't'+str(config.NUM_TRAINING_TRACES),
        #                       'T'+str(config.NUM_TESTING_TRACES),
        #                       'D' + str(config.GLOVE_OPTIONS['packetSize']),
        #                       'E' + str(config.GLOVE_OPTIONS['burstSize']),
        #                       'F' + str(config.GLOVE_OPTIONS['burstTime']),
        #                       'G' + str(config.GLOVE_OPTIONS['burstNumber']),
        #                       'H' + str(config.GLOVE_OPTIONS['biBurstSize']),
        #                       'I' + str(config.GLOVE_OPTIONS['biBurstTime']),
        #                       'A' + str(int(config.IGNORE_ACK)),
        #                       'V' + str(int(config.FIVE_NUM_SUM)),
        #                       'P' + str(int(config.n_components_PCA)),
        #                       'G' + str(int(config.n_components_LDA)),
        #                       'l' + str(int(config.lasso)),
        #                       'b' + str(int(config.bucket_Size))
#
        #                      ]

        #outputFilenameArray = arffFileName.split(config.RUN_ID)[1].split("-train")[0]
        # datafile-0dvnu5c1k80.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.5.b600-train.arff
        arffFileName = arffFileName.split("datafile-")[1]
        #outputFilenameArray = 'resultsRegression.' + arffFileName[25:-11]
        outputFilenameArray = 'resultsRegression.' + arffFileName[8:-11]

        outputFilename = os.path.join(config.OUTPUT_DIR,outputFilenameArray)

        if not os.path.exists(outputFilename+'.output'):
            banner = ['#All Features','Threshold','#Selected Features','Accuracy-All','Accuracy-Selected']
            f = open( outputFilename+'.output', 'w' )
            f.write(','.join(banner))
            f.close()

        return outputFilename

    @staticmethod
    def calcKmeans(files, numMonitored, numClusters, description):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []
        unmonClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                 currY = line.split(",")[-1]
                 if monClasses.__contains__(currY):  # add all testing monitored instances
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(line.split(",")[-1])
                 else: # nonMonitored instance
                     if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                         unmonClasses.append(currY)
                         X_testing.append([float(i) for i in line.split(",")[:-1]])
                         Y_testing.append(currY)

            #if line[0] != '@':
            #    X_testing.append([float(i) for i in line.split(",")[:-1]])
            #    Y_testing.append(line.split(",")[-1])

        #print instancesList

        X = np.array(trainingInstancesList)

        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA

        km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros
        #print radius
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            maxDist = -1
            for i in pointsIndex:
                # Euclidean distance
                currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
                if currDist > maxDist:
                    radius[clusIndx] = currDist
                #maxDist = currDist
                    maxDist = currDist



        #X_testing = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])

        #Y_testing = np.array([0,
        #              0,
        #              0,
        #              1,
        #              1,
        #              1])

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        for i in range(len(X_testing)):
            inside = False
            for clusIndx in range(len(km.cluster_centers_)):
               dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
               if dist <= radius[clusIndx]: #/1.5:#/2.0:
                   inside = True


            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1

        print "\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "NumClusters: " + str(numClusters)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)

        tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
        print "tpr = " + tpr
        print "fpr = " + fpr
        print "Acc = " + Acc
        print "F2  = " + F2

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(description)
        output.append(numClusters)
        output.append(config.RUN_ID)


        summary = '\t, '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileNameOW(files[0])

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''

    @staticmethod
    def calcEnsembleKmeans(files, numMonitored, numClusters, description):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []
        unmonClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                 currY = line.split(",")[-1]
                 if monClasses.__contains__(currY):  # add all testing monitored instances
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(line.split(",")[-1])
                 else: # nonMonitored instance
                     if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                         unmonClasses.append(currY)
                         X_testing.append([float(i) for i in line.split(",")[:-1]])
                         Y_testing.append(currY)

            #if line[0] != '@':
            #    X_testing.append([float(i) for i in line.split(",")[:-1]])
            #    Y_testing.append(line.split(",")[-1])

        #print instancesList

        X = np.array(trainingInstancesList)

        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA

        km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros
        #print radius
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            maxDist = -1
            for i in pointsIndex:
                # Euclidean distance
                currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
                if currDist > maxDist:
                    radius[clusIndx] = currDist
                #maxDist = currDist
                    maxDist = currDist



        #X_testing = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])

        #Y_testing = np.array([0,
        #              0,
        #              0,
        #              1,
        #              1,
        #              1])

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        for i in range(len(X_testing)):
            inside = False
            for clusIndx in range(len(km.cluster_centers_)):
               dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
               if dist <= radius[clusIndx]: #/1.5:#/2.0:
                   inside = True


            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1

        print "\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "NumClusters: " + str(numClusters)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)

        tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
        print "tpr = " + tpr
        print "fpr = " + fpr
        print "Acc = " + Acc
        print "F2  = " + F2

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(description)
        output.append(numClusters)
        output.append(config.RUN_ID)


        summary = '\t, '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileNameOW(files[0])

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''

    @staticmethod
    def calcKmeansCvxHullDelaunay(files, numMonitored, numClusters, description):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []
        unmonClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                 currY = line.split(",")[-1]
                 if monClasses.__contains__(currY):  # add all testing monitored instances
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(line.split(",")[-1])
                 else: # nonMonitored instance
                     if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                         unmonClasses.append(currY)
                         X_testing.append([float(i) for i in line.split(",")[:-1]])
                         Y_testing.append(currY)

            #if line[0] != '@':
            #    X_testing.append([float(i) for i in line.split(",")[:-1]])
            #    Y_testing.append(line.split(",")[-1])



        #print "X_testing length: " + str(len(X_testing))
        #print "Y_testing length: " + str(len(Y_testing))
        #print instancesList

        X = np.array(trainingInstancesList)

        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        # preprocessing, normalizing
        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before CPA

        km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

        hull = []

        #print radius
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
            hull.append(Delaunay(clusterPoints))


        #X_testing = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])

        #Y_testing = np.array([0,
        #              0,
        #              0,
        #              1,
        #              1,
        #              1])

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        for i in range(len(X_testing)):
            inside = False
            for hullIndx in range(len(hull)):
               if Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                   inside = True


            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1


        print "\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "NumClusters: " + str(numClusters)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)

        tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
        print "tpr = " + tpr
        print "fpr = " + fpr
        print "Acc = " + Acc
        print "F2  = " + F2

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(description)
        output.append(numClusters)


        summary = '\t, '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileNameOW(files[0])

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''

    @staticmethod
    def calcKmeansCvxHullDelaunay_Mixed(files, numMonitored, numClusters, description):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []
        unmonClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                 currY = line.split(",")[-1]
                 if monClasses.__contains__(currY):  # add all testing monitored instances
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(line.split(",")[-1])
                 else: # nonMonitored instance
                     if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                         unmonClasses.append(currY)
                         X_testing.append([float(i) for i in line.split(",")[:-1]])
                         Y_testing.append(currY)

            #if line[0] != '@':
            #    X_testing.append([float(i) for i in line.split(",")[:-1]])
            #    Y_testing.append(line.split(",")[-1])



        #print "X_testing length: " + str(len(X_testing))
        #print "Y_testing length: " + str(len(Y_testing))
        #print instancesList

        X = np.array(trainingInstancesList)

        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        # preprocessing, normalizing
        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before PCA
        #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before PCA

        km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            maxDist = -1
            for i in pointsIndex:
                # Euclidean distance
                currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
                if currDist > maxDist:
                    radius[clusIndx] = currDist
                #maxDist = currDist
                    maxDist = currDist



        hull = []

        fewPointsClusters = [] # indexes of clusters where there are < 12 points (convex hull needs 12 points)

        #print radius
        #http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.Delaunay.html
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
            if len(clusterPoints) >= 12:
                try:
                    hull.append(Delaunay(clusterPoints))#,qhull_options="C-0"))
                except:
                    print "Convex Hull ERROR"
                    description += "Convex Hull ERROR"
                    print " Cluster # " + str(clusIndx) + " -- Convex Hull ERROR. Kmeans cluster is to be checked for the participating points."
                    fewPointsClusters.append(clusIndx)
                    pass
            else:
                print " Cluster # " + str(clusIndx) + " doesn't have enough points to build a hull. Kmeans cluster is to be checked for the participating points."
                fewPointsClusters.append(clusIndx)


        #X_testing = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])

        #Y_testing = np.array([0,
        #              0,
        #              0,
        #              1,
        #              1,
        #              1])

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        #looping over mixed (hulls + not-enough-point clusters)
        for i in range(len(X_testing)):
            inside = False

            # looping over hulls
            for hullIndx in range(len(hull)):
                if Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                    inside = True

            # looping over not-enough-point clusters
            for clusIndx in range(len(km.cluster_centers_)):
                if fewPointsClusters.__contains__(clusIndx):
                    #print " Cluster # " + str(clusIndx) + " is being examined."
                    dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
                    if dist <= radius[clusIndx]: #/1.5:#/2.0:
                        inside = True

            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1


        print "\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "NumClusters: " + str(numClusters)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)

        tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
        print "tpr = " + tpr
        print "fpr = " + fpr
        print "Acc = " + Acc
        print "F2  = " + F2

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(description)
        output.append(numClusters)
        output.append(config.RUN_ID)


        summary = '\t, '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileNameOW(files[0])

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''

    @staticmethod
    def calcKmeansCvxHullDelaunay_Mixed_KNN(files, numMonitored, numClusters, description, threshold):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []
        unmonClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                 currY = line.split(",")[-1]
                 if monClasses.__contains__(currY):  # add all testing monitored instances
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(line.split(",")[-1])
                 else: # nonMonitored instance
                     if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                         unmonClasses.append(currY)
                         X_testing.append([float(i) for i in line.split(",")[:-1]])
                         Y_testing.append(currY)

            #if line[0] != '@':
            #    X_testing.append([float(i) for i in line.split(",")[:-1]])
            #    Y_testing.append(line.split(",")[-1])



        #print "X_testing length: " + str(len(X_testing))
        #print "Y_testing length: " + str(len(Y_testing))
        #print instancesList

        X = np.array(trainingInstancesList)

        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        # preprocessing, normalizing
        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before PCA
        #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before PCA

        km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            maxDist = -1
            for i in pointsIndex:
                # Euclidean distance
                currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
                if currDist > maxDist:
                    radius[clusIndx] = currDist
                #maxDist = currDist
                    maxDist = currDist



        hull = []

        fewPointsClusters = [] # indexes of clusters where there are < 12 points (convex hull needs 12 points)

        #print radius
        #http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.Delaunay.html
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
            if len(clusterPoints) >= 12:
                try:
                    hull.append(Delaunay(clusterPoints))#,qhull_options="C-0"))
                except:
                    print "Convex Hull ERROR"
                    description += " Convex Hull ERROR"
                    print " Cluster # " + str(clusIndx) + " -- Convex Hull ERROR. Kmeans cluster is to be checked for the participating points."
                    fewPointsClusters.append(clusIndx)
                    pass
            else:
                print " Cluster # " + str(clusIndx) + " doesn't have enough points to build a hull. Kmeans cluster is to be checked for the participating points."
                fewPointsClusters.append(clusIndx)


        #X_testing = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])

        #Y_testing = np.array([0,
        #              0,
        #              0,
        #              1,
        #              1,
        #              1])

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        #looping over mixed (hulls + not-enough-point clusters)
        for i in range(len(X_testing)):

            inside = False

            # looping over hulls
            for hullIndx in range(len(hull)):
                if inside != True and Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                    inside = True

                # KNN
                if inside != True and Utils.is_knn_to_hull_border_points(X, X_testing[i],hull[hullIndx],threshold):
                    inside = True

            # looping over not-enough-point clusters
            if inside != True:
                for clusIndx in range(len(km.cluster_centers_)):
                    if fewPointsClusters.__contains__(clusIndx):
                        #print " Cluster # " + str(clusIndx) + " is being examined."
                        dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
                        if dist <= radius[clusIndx]: #/1.5:#/2.0:
                            inside = True

            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1


        print "\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "NumClusters: " + str(numClusters)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)

        tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
        print "tpr = " + tpr
        print "fpr = " + fpr
        print "Acc = " + Acc
        print "F2  = " + F2

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(description)
        output.append(numClusters)
        output.append(config.RUN_ID)


        summary = '\t, '.join(itertools.imap(str, output))

        outputFilename = Utils.getOutputFileNameOW(files[0])

        f = open( outputFilename+'.output', 'a' )
        f.write( "\n"+summary )
        f.close()

        print ''

    @staticmethod
    def calcKmeansCvxHullDelaunay_Testing(files, numMonitored):

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        trainingInstancesList = []
        monClasses = []

        X_testing=[]
        Y_testing=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                X_testing.append([float(i) for i in line.split(",")[:-1]])
                Y_testing.append(line.split(",")[-1])

        #print instancesList

        #X = np.array(trainingInstancesList)
        X = np.random.rand(30, 2)
        #X = np.array([[1, 2],
        #              [5, 8],
        #              [1.5, 1.8],
        #              [8, 8],
        #              [1, 0.6],
        #              [9, 11]])
        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA

        km = KMeans(n_clusters=numMonitored, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

        km.fit(X) # building the clusters from the monitored instances only

        #print km.cluster_centers_[0]

        #print km.labels_

        # indexes of point in a specific cluster
        #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

        #print index

        # get radius of each cluster
        radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

        hull = []

        #print radius
        for clusIndx in range(len(km.cluster_centers_)):
            # indexes of points in a specific cluster
            pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
            clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
            hull.append(Delaunay(clusterPoints))


        X_testing = np.random.rand(30, 2)

        Y_testing = np.random.rand(30, 1)

        #monClasses = [0,1]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        inside = False

        for i in range(len(X_testing)):
            inside = False
            for hullIndx in range(len(hull)):
               if Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                   inside = True


            if inside:
               if monClasses.__contains__(Y_testing[i]):
                   tp += 1
               else:
                   fp += 1
            else:
               if monClasses.__contains__(Y_testing[i]):
                   fn += 1
               else:
                   tn += 1

        print "\n\n"
        print "radii: "
        print radius
        print "NumMonitored: " + str(numMonitored)
        print "dataset: " + str(files)

        print "tp = " + str(tp)
        print "tn = " + str(tn)
        print "fp = " + str(fp)
        print "fn = " + str(fn)


        print "tpr = " + str( float(tp)/float(tp+fn) )
        print "fpr = " + str( float(fp)/float(fp+tn) )
        print "Acc = " + str( float(tp+tn)/float(tp+tn+fp+fn) )
        print "F2  = " + str( float(5*tp)/float((5*tp)+(4*fn)+(fp)) ) # beta = 2

    @staticmethod
    def calcWeightsLogisticRegressionTest():

        boston = load_boston()
        X, y = boston['data'], boston['target']



        #print 'X'
        #print X
        #print '\n'
        #print 'y'
        #print y
#
        X = X[0:20,0:5]
        y = y[0:20]
        y = np.array(y).astype(np.int)
#
        print 'X'
        print X
        print '\n'
        print 'y'
        print y

        # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
        clf = LogisticRegression(penalty='l2',max_iter=1000,solver='newton-cg') # solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}
        #clf = LogisticRegression()
        #clf = LassoCV(max_iter=1000)

        # Set a minimum threshold of 0.25
        #sfm = SelectFromModel(clf, threshold=0.25)
        #sfm = SelectFromModel(clf, threshold=0)

        clf.fit(X, y)

        print "Weights (coef)"
        print clf.coef_
        print clf.__dict__


        print "\n"




    @staticmethod
    def calcPCA_ow(files):

        trainList = Utils.readFile1(files[0])
        testList = Utils.readFile1(files[1])

        instancesListTrain = []
        instancesListTest = []
        classesTrain = ""
        classesTest = ""
        yTrain=[]
        yTest=[]

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classesTrain = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesListTrain.append([float(i) for i in line.split(",")[:-1]])
                yTrain.append(line.split(",")[-1])

        for line in testList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classesTest = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesListTest.append([float(i) for i in line.split(",")[:-1]])
                yTest.append(line.split(",")[-1])
            #if line[0] != '@':
            #    instancesListTest.append([float(i) for i in line.split(",")[:-1]])
            #    yTest.append(line.split(",")[-1])

        #print instancesList

        XTrain = np.array(instancesListTrain)
        XTest = np.array(instancesListTest)

        #print X

        #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
        #preprocessing.scale(XTrain, axis=0, with_mean=True, with_std=True, copy=False)
        #preprocessing.scale(XTest, axis=0, with_mean=True, with_std=True, copy=False)
        #print X
        #config.n_components_PCA = 39

        pca = PCA(n_components=config.n_components_PCA)
        #sklearn_transf = pca.fit(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        sklearn_transf = pca.fit(XTrain)

        #X_transformed = sklearn_transf.transform(X[0:config.BUCKET_SIZE * config.NUM_TRAINING_TRACES])
        XTrain_transformed = sklearn_transf.transform(XTrain)
        XTest_transformed = sklearn_transf.transform(XTest)

        #print X_transformed[0:1,0:5]
        #print "EigVec", sklearn_transf.components_[0]
        #print sklearn_transf
        #print pca.explained_variance_

        #print pca.explained_variance_ratio_

        print ('\nPCA explained variance (first %d components): %.2f'%(config.n_components_PCA, sum(pca.explained_variance_ratio_)))

        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION sites')
        newTestList.append('@RELATION sites')

        for i in range(0,config.n_components_PCA):
            newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
            newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

        newTrainList.append('@ATTRIBUTE class '+classesTrain)
        newTestList.append('@ATTRIBUTE class '+classesTest)

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        ctr = 0

        #for item in X_lda_sklearn:
        for i in xrange(len(XTrain_transformed)):
            webpageInstance = XTrain_transformed[i]
            newTrainList.append(','.join([str("%.2f" % k) for k in webpageInstance]) + ',' + yTrain[i])

        for i in xrange(len(XTest_transformed)):
            webpageInstance = XTest_transformed[i]
            newTestList.append(','.join([str("%.2f" % k) for k in webpageInstance]) + ',' + yTest[i])


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_PCA_'+str(config.n_components_PCA)+'.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

    @staticmethod
    def calcCvxHull(points):
        from scipy.spatial import ConvexHull
        #points = np.random.rand(30, 2)   # 30 random points in 2-D
        hull = ConvexHull(points)

        import matplotlib.pyplot as plt
        plt.plot(points[:,0], points[:,1], 'o')

        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
        plt.show()

    @staticmethod
    def calcCvxHull_Delaunay(points):
        from scipy.spatial import Delaunay
        #points = np.random.rand(30, 2)   # 30 random points in 2-D
        hull = Delaunay(points)

        import matplotlib.pyplot as plt
        plt.plot(points[:,0], points[:,1], 'o')

        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
        plt.show()

    @staticmethod
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0


    @staticmethod
    def is_knn_to_hull_border_points(X, p, hull, threshold):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        # gather indecies from the convex_hull function
        border_points_indecies = []
        for item in hull.convex_hull:
            for index in item:
                #print index
                if not border_points_indecies.__contains__(index):
                    border_points_indecies.append(index)

        temp_dist_list = []
        for i in border_points_indecies: #.simplices: # simplex is the border points index
            #print X[i]
            dist = np.linalg.norm(p - X[i])
            #print dist
            temp_dist_list.append(dist)
            if dist <= threshold:
                print "returning true KNN for dist: " + str(dist)
                return True

        #print "avg of dist: "
        #print reduce(lambda x, y: x + y, temp_dist_list) / len(temp_dist_list)

        #print "sorted dist list: "
        #print sorted(temp_dist_list)

        return False


    @staticmethod
    def plotCvxHull_Delaunay(points,hull):
        from scipy.spatial import Delaunay
        #points = np.random.rand(30, 2)   # 30 random points in 2-D
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        import matplotlib.pyplot as plt
        plt.plot(points[:,0], points[:,1], 'o')

        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
        plt.show()

    @staticmethod
    def plotKmeans(points,centroid):
         pass

    @staticmethod
    def getOutputFileNameOW(arffFileName):
        # arffFileName
        # datafile-openworld5.i2vt8sjxk300.c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff
        arffFileName = arffFileName.split("datafile-openworld")[1]
        arffFileName = arffFileName.split("-train")[0]

        # arffFileName
        # 5.i2vt8sjxk300.c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff
        numMonitored = arffFileName.split(".")[0] # 5
        if len(numMonitored) == 1:
            arffFileName = arffFileName[14:]
        elif len(numMonitored) == 2:
            arffFileName = arffFileName[15:]
        else:
            arffFileName = arffFileName[16:] # three or more

        # arffFileName
        # .c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff
        #arffFileName = arffFileName[:-11]

        # arffFileName
        # .c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600

        outputFilename = 'resultsOW.m' + str(numMonitored) + arffFileName

        outputFilename = os.path.join(config.OUTPUT_DIR,outputFilename)

        if not os.path.exists(outputFilename+'.output'):
            banner = ['tpr','fpr','Acc','F2','tp','tn','fp','fn','description','num Clusters / cvx Hulls','File ID']
            f = open( outputFilename+'.output', 'w' )
            f.write('\t, '.join(banner))
            f.close()

        return outputFilename


    @staticmethod
    def testDelaunay():
        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()

        print tri.simplices # [[3 2 0] [3 1 0]]
        print tri.neighbors # [[-1  1 -1] [-1  0 -1]]
        print tri.convex_hull # [[2 0]
                              #  [3 2]
                              #  [1 0]
                              #  [3 1]]


        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], [0.5, 0.5]])
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()

        print tri.simplices # [[4 2 0]
                            # [4 1 0]
                            # [3 4 2]
                            # [3 4 1]]
        print tri.neighbors # [[-1  1  2]
                            # [-1  0  3]
                            # [ 0 -1  3]
                            # [ 1 -1  2]]
        print tri.convex_hull # [[2 0]
                              #   [1 0]
                              #   [3 2]
                              #   [3 1]]

        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], [0.5, 0.5], [0.25, 0.25]])
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()

        print tri.simplices #
        print tri.neighbors #
        print tri.convex_hull # [[2 0]
                              #   [1 0]
                              #   [3 2]
                              #   [3 1]]

        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], [0.5, 0.5], [0.25, 0.25], [1.5, 0.5]])
        tri = Delaunay(points)
        plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()

        print tri.simplices #
        print tri.neighbors #
        print tri.convex_hull # [[2 0]
                              #   [1 0]
                              #   [3 2]
                              #   [3 1]]
        border_points_indecies = []
        for item in tri.convex_hull:
            for index in item:
                print index
                if not border_points_indecies.__contains__(index):
                    border_points_indecies.append(index)

        print border_points_indecies

        #pass



    @staticmethod
    def goodSample(webpageIds, traceIndexStart, traceIndexEnd, threshold, numPackets):

        from Datastore import Datastore

        badWebsites = {} # <website, count of bad traces>

        for webpageId in webpageIds:
            webpage = Datastore.getWebpagesLL( [webpageId], traceIndexStart, traceIndexEnd )
            webpage = webpage[0]
            for trace in webpage.getTraces():
                #print "packet count:" + str(trace.getPacketCount())
                #print "Webpage ID: " + str(trace.getId())
                #print 'out'
                #print trace.getPacketCount()
                if trace.getPacketCount() < numPackets:
                    #print 'in'
                    #print trace.getPacketCount()
                    #print
                    #print "Webpage ID: " + str(trace.getId())
                    dataKey = str(trace.getId())
                    if not badWebsites.get( dataKey ):
                        badWebsites[dataKey] = 0

                    # keep the number of bad traces (packetCount<numPackets passed) for each website
                    badWebsites[dataKey] += 1
                    #return False

        if Utils.isBadSample(badWebsites, threshold):
            return False

        return True

    @staticmethod
    def isBadSample(badWebsites, threshold):
        #print "Bad websites:"
        #print badWebsites
        #print "\n"
        isBad = False
        for webKey, webValue in badWebsites.items():
            if webValue > threshold:
                config.BAD_WEBSITES.append(int(webKey))
                isBad = True

        return isBad



    @staticmethod
    def calcTPR_FPR(debugInfo, outputFilename, positive, negative):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for entry in debugInfo:
            if entry[0] in positive: # actual is positive
                if entry[1] in positive: # predicted is positive too
                    tp += 1
                else: # predicted is negative
                    fn += 1
            elif entry[0] in negative: # actual is negative
                if entry[1] in positive: # predicted is positive
                    fp += 1
                else: # predicted is negative too
                    tn += 1

        tpr = str( "%.4f" % (float(tp)/float(tp+fn)) )
        fpr = str( "%.4f" % (float(fp)/float(fp+tn) ))
        Acc = str( "%.4f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
        F2  = str( "%.4f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2

        if not os.path.exists(outputFilename+'.binary'):
            banner = ['tpr','fpr','Acc','F2','tp','tn','fp','fn','File ID']
            f = open( outputFilename+'.binary', 'w' )
            f.write('\t, '.join(banner))
            f.close()

        output = []
        output.append(tpr)
        output.append(fpr)
        output.append(Acc)
        output.append(F2)
        output.append(str(tp))
        output.append(str(tn))
        output.append(str(fp))
        output.append(str(fn))
        output.append(config.RUN_ID)

        summary = '\t, '.join(itertools.imap(str, output))

        f = open( outputFilename+'.binary', 'a' )
        f.write( "\n"+summary )
        f.close()

    @staticmethod
    def drawROC_AUC(debugInfo, positive, negative):
        actual = []
        predictions = []
        # binarize
        for entry in debugInfo:
            if entry[0] in positive:
                actual.append(1)
            else:
                actual.append(0)
            if entry[1] in positive:
                predictions.append(1)
            else:
                predictions.append(0)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # Joining two training and testing arff file into a single one
    @staticmethod
    def joinTrainingTestingFiles(trainFileName, testFileName):
        trainFileLines = [line.strip() for line in open(trainFileName)]
        testFileLines = [line.strip() for line in open(testFileName)]
        headersList = []
        instancesList = []

        for line in testFileLines: # as classes has mon and nonMon, in case of open world
            if line[0] == '@':
                headersList.append(line)

        for line in trainFileLines:
            if line[0] != '@':
                instancesList.append(line)

        for line in testFileLines:
            if line[0] != '@':
                instancesList.append(line)

        random.shuffle(instancesList)
        random.shuffle(instancesList)

        outputFile = trainFileName[:-11] + '.arff'

        f = open( outputFile, 'w' )
        f.write( "\n".join( headersList ) )
        f.write( "\n" )
        f.write( "\n".join( instancesList ) )
        f.close()

        return outputFile

    @staticmethod
    def plotDensity(files):

        trainList = Utils.readFile(files[0])

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2].split("{")[1].split("}")[0].split(",")

        #cmap = plt.get_cmap('gnuplot')
        #colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
        colors = ['r', 'b', 'g', 'k', 'm']

        currWebsite = classes[0]
        prevWebsite = currWebsite
        iColor = 0
        for line in trainList:
            if line[0] != '@':
                #print line
                currWebsite = line.split(",")[-1]
                instance = [float(i) for i in line.split(",")[:-1]]
                data = instance
                density = gaussian_kde(data)
                xs = np.linspace(0,5)
                density.covariance_factor = lambda : .25
                density._compute_covariance()

                if currWebsite != prevWebsite:
                    iColor += 1
                    if iColor == len(colors):
                        iColor = 0

                plt.plot(xs,density(xs),color=colors[iColor],label=currWebsite)

                prevWebsite = currWebsite


        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('Fingerprints of ' + str(len(classes)) + ' websites')

        plt.show()


    @staticmethod
    def plot (files):

        trainList = Utils.readFile(files[0])

        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2].split("{")[1].split("}")[0].split(",")

        #cmap = plt.get_cmap('gnuplot')
        #colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
        colors = ['r', 'b', 'g', 'k', 'm']

        currWebsite = classes[0]
        prevWebsite = currWebsite
        iColor = 0
        for line in trainList:
            if line[0] != '@':
                #print line
                currWebsite = line.split(",")[-1]
                instance = [float(i) for i in line.split(",")[:-1]]
                data = instance
                #density = gaussian_kde(data)
                ys = data
                #xs = np.linspace(0,5)
                xs = range(0,len(instance))
                #density.covariance_factor = lambda : .25
                #density._compute_covariance()

                if currWebsite != prevWebsite:
                    iColor += 1
                    if iColor == len(colors):
                        iColor = 0

                plt.plot(xs,ys,color=colors[iColor],label=currWebsite)

                prevWebsite = currWebsite


        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title('Fingerprints of ' + str(len(classes)) + ' websites')

        plt.show()

    @staticmethod
    def kNN():
        X = [[0], [1], [2], [3]]
        y = [0, 0, 1, 1]
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        print(neigh.predict([[1.1]]))

        print(neigh.predict_proba([[0.9]]))

    @staticmethod
    def kNN_mydist():
        #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2], [-3, -2], [-3, -2]])
        #y = np.array([[0], [0], [0], [1], [1], [1]])
        y = np.array([[0], [0], [0], [1], [1], [1], [0], [0]])
        from sklearn.neighbors import KNeighborsClassifier
        #neigh = KNeighborsClassifier(n_neighbors=3, metric=Utils.mydist)
        neigh = KNeighborsClassifier(metric=Utils.mydist)
        print X
        print y
        neigh.fit(X, y)
        print "here"
        print(neigh.predict([[1.1, 2]]))

        #print(neigh.predict_proba([[0.9, 1]]))

    @staticmethod
    def mydist(x, y):
        print np.sum((x-y)**2)
        return np.sum((x-y)**2)

    @staticmethod
    def kNN_mydist_lcs():
        data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
        X = np.arange(len(data)).reshape(-1, 1) # an index into a separate data structure (data)
        y = np.array([[0], [0], [0], [1], [1], [1]])

        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3, metric=Utils.mydist_lcs, algorithm='brute') # kNN distance metric overwritten, see LCS_dist. algorithm='brute' is a must
        neigh.fit(X, y)
        testing = np.array([[2,1,6,5,4,4],[2,1,6,5,4,4]])
        print(neigh.predict(np.arange(len(testing)).reshape(-1, 1)))

        #print(neigh.predict_proba([[0.9, 1]]))

    @staticmethod
    def mydist_lcs(xx, X): # xx is the testing instance, X is the training instance (passed one by one)
        data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
        testing = np.array([[2,1,6,5,4,4],[2,1,6,5,4,4]])
        import mlpy

        i, j = int(xx[0]), int(X[0])     # extract indices
        '''
        print i
        print j
        print testing[i]
        print data[j]
        '''
        length, path = mlpy.lcs_std(testing[i], data[j])

        dist_lcs = float(length)/np.sqrt(len(testing[i])*len(data[j])) ## 4.1.2 in Anomaly Detection for Discrete Sequences: A Survey
        dist_lcs_inv = float(1/float(dist_lcs))
        print str(dist_lcs) + ", " + str(dist_lcs_inv)
        return dist_lcs_inv



    data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4], [2,1,6,5,4],[2,3,6,5,4,4],[2,3,6,5,4,4],[2,3,6,5,4,4]])
    testing = np.array([[2,1,6,5,4,4],[2,1,6,5,4]])
    @staticmethod
    def calcHP_KNN_LCS():
        #data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
        X = np.arange(len(Utils.data)).reshape(-1, 1) # an index into a separate data structure (data)
        y = np.array([[0], [0], [0], [1], [1], [1], [1], [0], [0], [0]])
        #X = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        #y = [0,0,0,1,1,1,1,0,0,0]
        print X
        print y
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3, metric=Utils.LCS_dist, algorithm='brute') # kNN distance metric overwritten, see LCS_dist. algorithm='brute' is a must
        neigh.fit(X, y)
        #testing = np.array([[2,1,6,5,4,4]])
        print "\n"
        print(neigh.predict(np.arange(len(Utils.testing)).reshape(-1, 1)))

        #print(neigh.predict_proba([[0.9, 1]]))

    @staticmethod
    def LCS_dist(xx, X): # xx is the testing instance, X is the training instance (passed one by one)
        #data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
        #testing = np.array([[2,1,6,5,4,4]])
        import mlpy

        i, j = int(xx[0]), int(X[0])     # extract indices
        print "i = " + str(i) + ", j = " +str(j)
        '''
        print i
        print j
        print testing[i]
        print data[j]
        '''
        length, path = mlpy.lcs_std(Utils.testing[i], Utils.data[j])

        dist_lcs = float(length)/np.sqrt(len(Utils.testing[i])*len(Utils.data[j])) ## 4.1.2 in Anomaly Detection for Discrete Sequences: A Survey
        dist_lcs_inv = float(1/float(dist_lcs))
        #print str(length) + ", " + str(dist_lcs) + ", " + str(dist_lcs_inv)
        return dist_lcs_inv




    data2 = [[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4], [2,1,6,5,4],[2,3,6,5,4,4],[2,3,6,5,4,4],[2,3,6,5,4,4]]
    testing2 = np.array([[2,1,6,5,4,4,-1,-1],[2,1,6,5,4,-1,-1,-1]])
    #testing2 = np.array([[2,1,6,5,4,4,-1,-1]])

    @staticmethod
    def calcHP_KNN_LCS_fixed():
        #data = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
        #X = np.arange(len(Utils.data)).reshape(-1, 1) # an index into a separate data structure (data)
        X = Utils.make_square(Utils.data2)
        #y = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        #y = np.array([[0], [0], [0], [1], [1], [1], [1], [0], [0], [0]])
        y = np.array([[0], [0], [0], [1], [1], [1], [1], [0], [0], [0]])
        #print X
        #print y
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3, metric=Utils.LCS_dist_fixed,algorithm='brute')
        #neigh = KNeighborsClassifier(n_neighbors=3,weights='distance',metric='pyfunc',func=Utils.LCS_dist_fixed)
        #neigh = KNeighborsClassifier(metric=Utils.LCS_dist_fixed)
        neigh.fit(X, y)

        #print "here"
        #print X
        print(neigh.predict(Utils.testing2))

        #print(neigh.predict_proba([[0.9, 1]]))

    @staticmethod
    def LCS_dist_fixed(xx, Xxx): # xx is the testing instance, X is the training instance (passed one by one)

        import mlpy

        length, path = mlpy.lcs_std(xx, Xxx)

        dist_lcs = float(length)/np.sqrt(len(xx)*len(Xxx)) ## 4.1.2 in Anomaly Detection for Discrete Sequences: A Survey
        dist_lcs_inv = float(1/float(dist_lcs))
        print str(length) + ", " + str(dist_lcs) + ", " + str(dist_lcs_inv)
        return dist_lcs_inv

    @staticmethod
    def make_square(jagged):
        # Careful: this mutates the series list of list
        max_cols = max(map(len, jagged))
        for row in jagged:
            row.extend([-1] * (max_cols - len(row)))

        #return np.array(jagged, dtype=np.float)

        return np.array(jagged)


#Utils.calcHP_KNN_LCS()
#Utils.kNN_mydist_lcs()
#Utils.calcHP_KNN_LCS_fixed()
#Utils.kNN_mydist()

    @staticmethod
    def calcTreeBaseFSTest():
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        iris = load_iris()
        X, y = iris.data, iris.target

        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        importances = clf.feature_importances_

        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


        #model = SelectFromModel(clf, prefit=True)
        #X_new = model.transform(X)
        #X_new.shape

        '''
        trainList = Utils.readFile(files[0])

        instancesList = []
        classes = ""
        y=[]
        for line in trainList:
            if line[0] == '@':
                 if line.lower().startswith("@attribute class"):
                     classes = line.split(" ")[2]
            else:
                #instancesList.append(float(line.split(",")[:-1]))
                instancesList.append([float(i) for i in line.split(",")[:-1]])
                #y.append(line.split(",")[-1])
                y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers
        pass
        '''

    @staticmethod
    def calcTreeBaseFSTest2():
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        iris = load_iris()
        X, y = iris.data, iris.target

        #print X.shape # (150, 4)
        # split X and y, 100 for training (feature selection) and 50 for testing
        Xtr = X[:100]
        ytr = y[:100]
        Xte = X[100:]
        yte = y[100:]

        clf = ExtraTreesClassifier()
        clf = clf.fit(Xtr, ytr)
        importances = clf.feature_importances_ # feature selection

        # model built from Xtr only and will be used for Xte. This is to eliminate bias.
        # threshold=0.01 to get all features first
        model = SelectFromModel(clf, prefit=True, threshold=0.01)

        Xtr_new = model.transform(Xtr)
        n_features = Xtr_new.shape[1] # shape gives (n_rows, n_columns)
        n_features_needed = 2
        while n_features > n_features_needed:
            model.threshold += 0.05
            Xtr_new = model.transform(Xtr)
            n_features = Xtr_new.shape[1]

        print model.threshold

        # model built from Xtr only and will be used for Xte. This is to eliminate bias.
        Xte_new = model.transform(Xte)

        print Xte_new.shape
        print Xte_new

        # get the most important features
        indices = np.argsort(importances)[::-1]
        print indices
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(n_features_needed):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    @staticmethod
    def calcTreeBaseRF(files, n_features_needed):
        # Feature selection

        trainList = Utils.readFile(files[0])
        testList = Utils.readFile(files[1])

        Xtr = []
        featuresBlockArff = []
        ytr=[]
        for line in trainList:
            if line[0] == '@':
                featuresBlockArff.append(line)
            else:
                Xtr.append([float(i) for i in line.split(",")[:-1]])
                ytr.append(line.split(",")[-1])

        Xte = []
        yte = []
        for line in testList:
            if line[0] != '@':
                Xte.append([float(i) for i in line.split(",")[:-1]])
                yte.append(line.split(",")[-1])

        Xtr = np.array(Xtr)
        ytr = np.array(ytr)

        Xte = np.array(Xte)
        yte = np.array(yte)

        clf = ExtraTreesClassifier(criterion='entropy')
        clf = clf.fit(Xtr, ytr)
        importances = clf.feature_importances_ # feature selection

        # model built from Xtr only and will be used for Xte. This is to eliminate bias.
        # threshold=0.01 to get all features first
        model = SelectFromModel(clf, prefit=True, threshold=0.01)

        Xtr_new = model.transform(Xtr)
        n_features = Xtr_new.shape[1] # shape gives (n_rows, n_columns)

        print 'num features: ' + str(n_features)

        if n_features == 0:
            print 'No feature selection applied, data may be too noisy!'
            return [files[0], files[1]]

        # Check if model has less features than needed
        if n_features <= n_features_needed:
            n_features_needed = n_features
        else:
            # pick the most important features (n_features_needed)
            while n_features > n_features_needed:
                model.threshold += 0.05
                Xtr_new = model.transform(Xtr)
                n_features = Xtr_new.shape[1]

                if n_features <= n_features_needed:
                    # this breaks the loop
                    n_features_needed = n_features

        print 'threshold: ' + str(model.threshold)
        # model built from Xtr only and will be used for Xte. This is to eliminate bias.
        Xte_new = model.transform(Xte)

        # get the most important features
        indices = np.argsort(importances)[::-1]

        print indices
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(n_features_needed):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


        newTrainList = []
        newTestList = []

        newTrainList.append('@RELATION traces')
        newTestList.append('@RELATION traces')

        # selected features
        for i in range(1,len(featuresBlockArff)-2): # excluding @ATTRIBUTE class and @DATA
            if i-1 in indices[:n_features_needed]:
                newTrainList.append(featuresBlockArff[i])
                newTestList.append(featuresBlockArff[i])

        newTrainList.append(featuresBlockArff[-2]) # @ATTRIBUTE class and @DATA
        newTestList.append( featuresBlockArff[-2])  # @DATA

        newTrainList.append('@DATA')
        newTestList.append('@DATA')

        for i in xrange(len(Xtr_new)):
            instance = Xtr_new[i]
            newTrainList.append(','.join([str("%.2f" % k) for k in instance]) + ',' + ytr[i])

        for i in xrange(len(Xte_new)):
            instance = Xte_new[i]
            newTestList.append(','.join([str("%.2f" % k) for k in instance]) + ',' + yte[i])


        # writing the new training file (with lower dimensions)
        fnewTrainName = files[0][:-5]+'_FS_RForest' + str(n_features_needed) + '.arff'
        fnewTrain = open(fnewTrainName, 'w')
        for item in newTrainList:
            fnewTrain.write(item+'\n')

        fnewTrain.close()

        # writing the new testing file (with lower dimensions)
        fnewTestName = files[1][:-5]+'_FS_RForest' + str(n_features_needed) + '.arff'
        fnewTest = open(fnewTestName, 'w')
        for item in newTestList:
            fnewTest.write(item+'\n')

        fnewTest.close()

        return [fnewTrainName,fnewTestName]

#Utils.calcTreeBaseFSTest2()
#Utils.calcTreeBaseRF(['/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-11Nov2015/cache/datafile-47yhu0bp-rf-train.arff','/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-11Nov2015/cache/datafile-47yhu0bp-rf-test.arff'],1)


    @staticmethod
    def getEnsembleAccuracy(debugInfos,CLASSIFIER_LIST):

        ensembleDebugInfo = []
        numClassifiers = len(debugInfos)

        numTestInst = len(debugInfos[CLASSIFIER_LIST[0]])
        for i in range(0,numTestInst):
            actualClass = debugInfos[CLASSIFIER_LIST[0]][i][0]
            ensemblePredictedClasses = []
            ensembleClassfierList = []
            for classifier in debugInfos: # loop over keys
                ensemblePredictedClasses.append(debugInfos[classifier][i][1]) # classifier is key in debugInfos, i is row number, 1 is second column
                ensemblePredictedClasses.append(debugInfos[classifier][i][2]) # confidence
                ensembleClassfierList.append(classifier)

            majorityVoteClass = Utils.getMajorityVoteClassConfidence(ensemblePredictedClasses,ensembleClassfierList)
            #majorityVoteClass = Utils.getMajorityVoteClass(ensemblePredictedClasses,ensembleClassfierList)
            ensembleDebugInfo.append([actualClass,majorityVoteClass])


        totalPredictions = 0
        totalCorrectPredictions = 0

        for item in ensembleDebugInfo:
            totalPredictions += 1.0

            if item[0] == item[1]:
                totalCorrectPredictions += 1.0

        ensembleAccuracy = totalCorrectPredictions / totalPredictions * 100.0
        #print ensembleAccuracy
        return [ensembleAccuracy,ensembleDebugInfo]


    @staticmethod
    def getMajorityVoteClass(ensemblePredictedClasses, ensembleClassfierList):
        from collections import Counter
        c = Counter(ensemblePredictedClasses)

        value, count = c.most_common()[0]
        print ensemblePredictedClasses
        print ensembleClassfierList
        print c
        if count != len(ensemblePredictedClasses): # majority don't agree (tie in case of two classifiers. needs to be modified for three classifiers)
            preferredValue = ensemblePredictedClasses[ensembleClassfierList.index(config.ENSEMBLE)]
            print preferredValue
            print "----------"
            return preferredValue

        print value
        print "----------"
        return value

    @staticmethod
    def getMajorityVoteClassConfidence(ensemblePredictedClasses, ensembleClassfierList):
        from collections import Counter
        c = Counter(ensemblePredictedClasses)

        value, count = c.most_common()[0]
        print ensemblePredictedClasses
        print ensembleClassfierList
        print c
        maxWeight = -1
        maxWeighIndx = 0
        #if count != len(ensemblePredictedClasses): # majority don't agree (tie in case of two classifiers. needs to be modified for three classifiers)
        if count < 2 or Utils.isfloat(value): # in case of ['webpate1',0.56,'webpage0',0.56] so value is 0.56 and count is 2 (0.56 is the confidence of svm, not a webpage!)
            #preferredValue = ensemblePredictedClasses[ensembleClassfierList.index(config.ENSEMBLE)]
            for i in range(1,len(ensemblePredictedClasses),2): # ensemblePredictedClasses=['webpate1',0.56,'webpage0',0.2]
                if float(ensemblePredictedClasses[i])>maxWeight:
                    maxWeight = float(ensemblePredictedClasses[i])
                    maxWeighIndx = i

            preferredValue = ensemblePredictedClasses[maxWeighIndx-1]
            print preferredValue
            print "----------"
            return preferredValue

        print value
        print "----------"
        return value



    @staticmethod
    def isfloat(value):
        try:
            float(value) # throws an exception if value is a string, for example
            return True
        except:
            return False



    @staticmethod
    def getOutputFileName(arffFileName):
        # arffFileName
        # datafile-openworld5.i2vt8sjxk300.c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff
        arffFileName = arffFileName.split("datafile-")[1]
        arffFileName = arffFileName[8:]
        arffFileName = arffFileName.split("-train")[0]

        outputFilename = arffFileName
        outputFilename = os.path.join(config.OUTPUT_DIR,outputFilename)
        outputFilename = outputFilename+'.'+config.CLUSTERING[config.CLUSTERING_METHOD]+'.binary'

        if not os.path.exists(outputFilename):
            banner = ['tpr','fpr','Acc','F2','tp','tn','fp','fn','description','num Clusters / cvx Hulls','File ID']
            f = open( outputFilename, 'w' )
            f.write('\t, '.join(banner))
            f.close()

        return outputFilename





    '''
debugInfos={3: [['webpage170', 'webpage170'], ['webpage170', 'webpage170'], ['webpage268', 'webpage268'], ['webpage268', 'webpage268'], ['webpage411', 'webpage411'], ['webpage411', 'webpage411'], ['webpage184', 'webpage184'], ['webpage184', 'webpage184'], ['webpage485', 'webpage485'], ['webpage485', 'webpage485'], ['webpage478', 'webpage478'], ['webpage478', 'webpage478'], ['webpage69', 'webpage69'], ['webpage69', 'webpage184'], ['webpage646', 'webpage646'], ['webpage646', 'webpage646'], ['webpage732', 'webpage732'], ['webpage732', 'webpage732'], ['webpage143', 'webpage143'], ['webpage143', 'webpage143']], 15: [['webpage170', 'webpage170'], ['webpage170', 'webpage170'], ['webpage268', 'webpage268'], ['webpage268', 'webpage268'], ['webpage411', 'webpage411'], ['webpage411', 'webpage411'], ['webpage184', 'webpage184'], ['webpage184', 'webpage184'], ['webpage485', 'webpage485'], ['webpage485', 'webpage485'], ['webpage478', 'webpage478'], ['webpage478', 'webpage478'], ['webpage69', 'webpage69'], ['webpage69', 'webpage69'], ['webpage646', 'webpage646'], ['webpage646', 'webpage646'], ['webpage732', 'webpage732'], ['webpage732', 'webpage732'], ['webpage143', 'webpage143'], ['webpage143', 'webpage143']], 23: [['webpage170', 'webpage170'], ['webpage170', 'webpage170'], ['webpage268', 'webpage268'], ['webpage268', 'webpage268'], ['webpage411', 'webpage411'], ['webpage411', 'webpage411'], ['webpage184', 'webpage184'], ['webpage184', 'webpage184'], ['webpage485', 'webpage485'], ['webpage485', 'webpage485'], ['webpage478', 'webpage478'], ['webpage478', 'webpage478'], ['webpage69', 'webpage69'], ['webpage69', 'webpage268'], ['webpage646', 'webpage646'], ['webpage646', 'webpage646'], ['webpage732', 'webpage732'], ['webpage732', 'webpage732'], ['webpage143', 'webpage143'], ['webpage143', 'webpage143']]}
CLASSIFIER_LIST = [3, 15, 23]
Utils.getEnsembleAccuracy(debugInfos, CLASSIFIER_LIST)
    '''
'''
config.ENSEMBLE = 43
debugInfos={43: [['webpage0', 'webpage1'], ['webpage0', 'webpage1'], ['webpage0', 'webpage1'], ['webpage0', 'webpage1'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage9', 'webpage9'], ['webpage9', 'webpage26'], ['webpage9', 'webpage9'], ['webpage9', 'webpage9'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage13', 'webpage24'], ['webpage13', 'webpage24'], ['webpage13', 'webpage24'], ['webpage13', 'webpage20'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage15', 'webpage14'], ['webpage15', 'webpage14'], ['webpage15', 'webpage14'], ['webpage15', 'webpage14'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage17', 'webpage17'], ['webpage17', 'webpage17'], ['webpage17', 'webpage15'], ['webpage17', 'webpage14'], ['webpage18', 'webpage18'], ['webpage18', 'webpage18'], ['webpage18', 'webpage24'], ['webpage18', 'webpage18'], ['webpage19', 'webpage20'], ['webpage19', 'webpage20'], ['webpage19', 'webpage20'], ['webpage19', 'webpage20'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage21', 'webpage21'], ['webpage21', 'webpage24'], ['webpage21', 'webpage21'], ['webpage21', 'webpage24'], ['webpage22', 'webpage22'], ['webpage22', 'webpage24'], ['webpage22', 'webpage24'], ['webpage22', 'webpage24'], ['webpage23', 'webpage24'], ['webpage23', 'webpage23'], ['webpage23', 'webpage22'], ['webpage23', 'webpage26'], ['webpage24', 'webpage24'], ['webpage24', 'webpage24'], ['webpage24', 'webpage24'], ['webpage24', 'webpage24'], ['webpage25', 'webpage24'], ['webpage25', 'webpage24'], ['webpage25', 'webpage25'], ['webpage25', 'webpage2'], ['webpage26', 'webpage24'], ['webpage26', 'webpage24'], ['webpage26', 'webpage24'], ['webpage26', 'webpage24'], ['webpage27', 'webpage23'], ['webpage27', 'webpage24'], ['webpage27', 'webpage24'], ['webpage27', 'webpage23'], ['webpage28', 'webpage24'], ['webpage28', 'webpage24'], ['webpage28', 'webpage29'], ['webpage28', 'webpage28'], ['webpage29', 'webpage24'], ['webpage29', 'webpage24'], ['webpage29', 'webpage24'], ['webpage29', 'webpage24']], 23: [['webpage0', 'webpage0'], ['webpage0', 'webpage0'], ['webpage0', 'webpage0'], ['webpage0', 'webpage0'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage1', 'webpage1'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage2', 'webpage2'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage3', 'webpage3'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage4', 'webpage4'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage5', 'webpage5'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage6', 'webpage6'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage7', 'webpage7'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage8', 'webpage8'], ['webpage9', 'webpage9'], ['webpage9', 'webpage9'], ['webpage9', 'webpage9'], ['webpage9', 'webpage9'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage10', 'webpage10'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage11', 'webpage11'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage12', 'webpage12'], ['webpage13', 'webpage29'], ['webpage13', 'webpage29'], ['webpage13', 'webpage29'], ['webpage13', 'webpage13'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage14', 'webpage14'], ['webpage15', 'webpage15'], ['webpage15', 'webpage15'], ['webpage15', 'webpage15'], ['webpage15', 'webpage15'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage16', 'webpage16'], ['webpage17', 'webpage17'], ['webpage17', 'webpage17'], ['webpage17', 'webpage17'], ['webpage17', 'webpage17'], ['webpage18', 'webpage18'], ['webpage18', 'webpage18'], ['webpage18', 'webpage18'], ['webpage18', 'webpage18'], ['webpage19', 'webpage19'], ['webpage19', 'webpage19'], ['webpage19', 'webpage19'], ['webpage19', 'webpage19'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage20', 'webpage20'], ['webpage21', 'webpage29'], ['webpage21', 'webpage29'], ['webpage21', 'webpage29'], ['webpage21', 'webpage29'], ['webpage22', 'webpage29'], ['webpage22', 'webpage29'], ['webpage22', 'webpage29'], ['webpage22', 'webpage29'], ['webpage23', 'webpage29'], ['webpage23', 'webpage29'], ['webpage23', 'webpage29'], ['webpage23', 'webpage29'], ['webpage24', 'webpage29'], ['webpage24', 'webpage29'], ['webpage24', 'webpage29'], ['webpage24', 'webpage29'], ['webpage25', 'webpage29'], ['webpage25', 'webpage29'], ['webpage25', 'webpage29'], ['webpage25', 'webpage29'], ['webpage26', 'webpage29'], ['webpage26', 'webpage29'], ['webpage26', 'webpage29'], ['webpage26', 'webpage29'], ['webpage27', 'webpage29'], ['webpage27', 'webpage29'], ['webpage27', 'webpage29'], ['webpage27', 'webpage29'], ['webpage28', 'webpage29'], ['webpage28', 'webpage29'], ['webpage28', 'webpage29'], ['webpage28', 'webpage29'], ['webpage29', 'webpage29'], ['webpage29', 'webpage29'], ['webpage29', 'webpage29'], ['webpage29', 'webpage29']]}
CLASSIFIER_LIST = [43,23]
Utils.getEnsembleAccuracy(debugInfos, CLASSIFIER_LIST)
'''


'''
def makeLabelNoise(files, noiseRatio):

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    Xtr = []
    featuresBlockArff = []
    ytr=[]
    for line in trainList:
        if line[0] == '@':
            featuresBlockArff.append(line)
        else:
            Xtr.append([float(i) for i in line.split(",")[:-1]])
            ytr.append(line.split(",")[-1]) # label


    ytrBen = ytr[ : config.NUM_BENIGN_CLASSES*config.NUM_TRAINING_TRACES]
    ytrAttk = ytr[config.NUM_BENIGN_CLASSES*config.NUM_TRAINING_TRACES : ]


    return [accuracy,debugInfo]
    '''

#print Utils.isfloat('webpage1')

