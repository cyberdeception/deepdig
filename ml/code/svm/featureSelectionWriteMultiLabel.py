
import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch
import numpy as np

def main():

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2/"
    #datafile-1o6tm98hk60.c0.d3.C22.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0-train.arff
    #datafile-1o6tm98hk60.c0.d3.C22.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0-test.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)
                outputFoldername = myfile[9:-11]
                outputFoldername = os.path.join(FIELS_PATH, outputFoldername)

                print trainingFilename
                print testingFilename
                print outputFoldername
                print '\n'

                __writeMultiLabelArff(trainingFilename, testingFilename, outputFoldername)


def  __writeMultiLabelArff(trainingFilename, testingFilename, outputFoldername):


#    if not os.path.exists(outputFoldername):
#        os.mkdir(outputFoldername)
#    else:
#        shutil.rmtree(outputFoldername) # delete and remake folder
#        os.mkdir(outputFoldername)

    trainList = Utils.readFile(trainingFilename)
    testList = Utils.readFile(testingFilename)

    for line in trainList:
        if line[0] == '@':
             if line.lower().startswith("@attribute class"):
                 classes = line.split(" ")[2].split("{")[1].split("}")[0].split(",") # list of classes

    #print classes
    labelMap = toMultiLabel(classes)

    newTrainList = []
    newTestList = []

    for line in trainList:
        if line[0] == '@':
            newTrainList.append(line)
        else:
            webpage = line.split(",")[-1]
            instanceWithoutLabel = ",".join(line.split(",")[:-1])
            newTrainList.append(instanceWithoutLabel + "," + ",".join(str(i) for i in labelMap[webpage]))

    noFearures = len(line.split(",")[:-1]) # excluding class name

    for line in testList:
        if line[0] == '@':
            newTestList.append(line)
        else:
            webpage = line.split(",")[-1]
            instanceWithoutLabel = ",".join(line.split(",")[:-1])
            newTestList.append(instanceWithoutLabel + "," + ",".join(str(i) for i in labelMap[webpage]))

    fnewTrainName = trainingFilename[:-5]+'_MultiLabeled'+'.arff'
    fnewTrain = open(os.path.join(outputFoldername, fnewTrainName), 'w')
    for item in newTrainList:
        fnewTrain.write(item+'\n')

    fnewTrain.close()

    # writing the new testing file (with lower dimensions)
    fnewTestName = testingFilename[:-5]+'_MultiLabeled'+'.arff'
    fnewTest = open(os.path.join(outputFoldername, fnewTestName), 'w')
    for item in newTestList:
        fnewTest.write(item+'\n')

    fnewTest.close()

    # Wrting multi labels map (class to multilabel)
    fmultiLabel = trainingFilename[:-10]+'_NumberOfFeatures_LabelMap.txt'
    fLabelMap = open(os.path.join(outputFoldername, fmultiLabel), 'w')

    fLabelMap.write('Number of features: ' + str(noFearures)+'\n\n\n')


    fLabelMap.write('Labeling: '+'\n')
    fLabelMap.write('-----------\n')
    fLabelMap.write('WebpageID: '+'label\n')

    for key in labelMap:
        fLabelMap.write(key+': '+",".join(str(i) for i in labelMap[key])+'\n')

    fLabelMap.close()



def toMultiLabel(classes):
    labelMap = {}

    for webpage in classes:
        index = classes.index(webpage)
        bitList = np.zeros(len(classes))
        #print 'bitList'
        #print bitList
        bitList[index] = 1
        #print 'bitList'
        #print bitList
        bitList = [int(i) for i in bitList] # convert from float to an int list (0.0 to 0)
        labelMap[webpage] = bitList
        #print labelMap[webpage]
        #print ",".join(str(i) for i in labelMap[webpage])
        #print ''

    return labelMap

if __name__ == "__main__":
    main()