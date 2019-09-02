
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
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/processing/supervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2Processing/datasets"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2MM/pythonCode/cache"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try3_d3/code_C23_d3/cache/"
    #datafile-fiqtk29xk20.c0.d0.C23.N755.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0-train.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)

                featuresFilename = myfile[:-10] + "features.arff"
                featuresFilename = os.path.join(FIELS_PATH, featuresFilename)

                featuresFilename_2 = myfile[:-10] + "features_2.arff"
                featuresFilename_2 = os.path.join(FIELS_PATH, featuresFilename_2)

                outputFoldername = myfile[9:-11]
                outputFoldername = os.path.join(FIELS_PATH, outputFoldername)

                dataSet = trainingFilename.split("datafile-")[1] # emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                config.RUN_ID = dataSet[:8]

                print trainingFilename
                print testingFilename
                print featuresFilename
                print outputFoldername
                print '\n'

                if NumWebsites == 0:
                    #Utils.calcLogisticRegression([trainingFilename, testingFilename])
                    __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername,featuresFilename)
                    __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername,featuresFilename_2)

                else:
                    # check file if file k matches the passed one
                    fileK = trainingFilename.split("datafile-")[1] # emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                    fileK = fileK[9:] # (w/o k)   30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff
                    fileK = fileK.split(".")[0] # 30

                    if NumWebsites == int(fileK):
                        __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername,featuresFilename)
                        __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername,featuresFilename_2)




def  __applyFeatureSelection(trainingFilename, testingFilename, outputFoldername, featuresFilename):


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
    AccAllFeatures = str(accuracy)

    trainList = Utils.readFile(trainingFilename)
    testList = Utils.readFile(testingFilename)
    featuresList = Utils.readFile(featuresFilename)
    if len(featuresList) > 0:
        featuresList = featuresList[0].split("{")[1].split("}")[0].split(",")
    else:
        featuresList = [0, 1, 2] # dummy features
    featuresList = [int(i) for i in featuresList]
    featuresList = sorted(featuresList)
    #print featuresList.__contains__(4)

    NumSelectedFeatures = len(featuresList)

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

    NumAllFeatures = len(instanceSplit) - 1

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

    AccSelectedFeatures = str(accuracy)

    output = [NumAllFeatures]
    output.append(NumSelectedFeatures)
    output.append(AccAllFeatures)
    output.append(AccSelectedFeatures)
    output.append(config.RUN_ID)

    writeOutputResultsFile(trainingFilename, output)

    print ''


#@staticmethod
def writeOutputResultsFile(arffFileName, output):
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

    outputFilenameArray = 'resultsMM.' + arffFileName[8:-11]

    outputFilename = os.path.join(config.OUTPUT_DIR,outputFilenameArray)

    if not os.path.exists(outputFilename+'.output'):
        banner = ['#All Features','#Selected Features','Accuracy-All','Accuracy-Selected','file ID']
        f = open( outputFilename+'.output', 'w' )
        f.write(','.join(banner))
        f.close()

    summary = ', '.join(itertools.imap(str, output))
    f = open( outputFilename+'.output', 'a' )
    f.write( "\n"+summary )
    f.close()

if __name__ == "__main__":
    main()