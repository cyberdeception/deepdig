
import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch
import numpy as np
import wekaAPI
import getopt
import sys

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression

def main():

    NumWebsites = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], "k:h") # k means number of website (to write scripts that pass -k 20, 40, ... to run multiple experiments in parallel
    except getopt.GetoptError, err:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o in ("-k"):
            NumWebsites = int(a)
        else:
            sys.exit(2)

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/WebsiteFingerprinting1/processing/upsupervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/processing/supervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2"
    FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-11Nov2015/cache"
    #datafile-emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff

    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)

                print trainingFilename
                print testingFilename
                print '\n'

                dataSet = trainingFilename.split("datafile-")[1] # emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                dataSet = dataSet[9:] # (w/o k)   30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                dataSet = dataSet.split("d")[1]
                dataSet = dataSet.split(".")[0]

                if int(dataSet) == 0:
                    config.Num_Features_Selected = 50
                else:
                    config.Num_Features_Selected = 200

                config.lasso = 0.1 #0.25 # threshold


                if NumWebsites == 0:
                    Utils.calcLogisticRegression([trainingFilename, testingFilename])
                else:
                    # check file if file k matches the passed one
                    fileK = trainingFilename.split("datafile-")[1] # emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                    fileK = fileK[9:] # (w/o k)   30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                    fileK = fileK.split(".")[0]

                    if NumWebsites == int(fileK):
                        Utils.calcLogisticRegression([trainingFilename, testingFilename])


                #featuresFilename = myfile[:-10] + "features_2.arff"
                #featuresFilename = os.path.join(FIELS_PATH, featuresFilename)

                #outputFoldername = myfile[9:-11]
                #outputFoldername = os.path.join(FIELS_PATH, outputFoldername)

                #print trainingFilename
                #print testingFilename
                #print featuresFilename
                #print outputFoldername
                #print '\n'

                #__applyFeatureSelection(trainingFilename, testingFilename, outputFoldername,selectedFeaturesList)

'''
def  __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername, featuresList):


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
    print outputFoldername
    print 'accuracy before feature selection ' + str(accuracy)

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
    fnewTrain = open(os.path.join(outputFoldername, fnewTrainName), 'w')
    for item in newTrainList:
        fnewTrain.write(item+'\n')

    fnewTrain.close()

    # writing the new testing file (with lower dimensions)
    fnewTestName = testingFilename[:-5]+'_Features'+'.arff'
    fnewTest = open(os.path.join(outputFoldername, fnewTestName), 'w')
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

    print ''


#-----------------

def calcLogisticRegression(files):

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
            #y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

    #for line in testList:
    #    if line[0] != '@':
    #        instancesList.append([float(i) for i in line.split(",")[:-1]])
    #        y.append(line.split(",")[-1])
    #        #y.append(line.split(",")[-1].split("webpage")[1]) # taking the ID of the website as the library works on numbers

    #print instancesList

    X = np.array(instancesList) #.astype(np.float)
    y = np.array(y)

    clf = LogisticRegression(penalty='l2',max_iter=1000,solver='newton-cg') # solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.25)
    #sfm = SelectFromModel(clf, threshold=0)

    sfm.fit(X, y) # take training data only
    n_features = sfm.transform(X).shape[1]

    #print 'n_features'
    #print n_features
    #print '\n'

    print '\n'
    print 'sfm.get_support'
    print sfm.get_support(indices=True)

    selectedFeaturesList = np.array(sfm.get_support(indices=True)).tolist()

    print selectedFeaturesList

    Utils.__applyFeatureSelection(files[0], files[1], selectedFeaturesList)

'''

if __name__ == "__main__":
    main()