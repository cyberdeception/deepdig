
import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch
import numpy as np
import subprocess

def main():

    mm_code_path = "/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/MM_Code/11-23/temp/"

    # 1- MultiLabeling
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2/"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try3_d3/code_C23_d3_FeatureSelectionAll_Testing/cache/"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try3_d3/code_C23_d3_FeatureSelectionAll_Testing/cache2/"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try3_d3/code_C23_d3/cache/"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/PanckenkoFeatureSelection/code_C3_d3/cache"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/testing"
    #datafile-5pbr2e6nk20.c0.d3.C23.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.0.b600-train.arff
    #datafile-5pbr2e6nk20.c0.d3.C23.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.0.b600-test.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)
                outputFoldername = myfile[9:-11]
                outputFoldername = os.path.join(FIELS_PATH, outputFoldername)

                #print trainingFilename
                #print testingFilename
                #print outputFoldername
                #print '\n'

                __writeMultiLabelArff(trainingFilename, testingFilename, outputFoldername) # has X_len (#features) and Y_len(#labels) in the MultiLabel file


    # 2- Create Feature files
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2MM/pythonCode/cache"
    #datafile-fiqtk29xk20.c0.d0.C23.N755.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0-train.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)

                featuresFilename = myfile[:-10] + "features.arff"
                featuresFilename = os.path.join(FIELS_PATH, featuresFilename)

                #if not os.path.exists(featuresFilename):
                f = open( featuresFilename, 'w' )
                f.close()

                featuresFilename_2 = myfile[:-10] + "features_2.arff"
                featuresFilename_2 = os.path.join(FIELS_PATH, featuresFilename_2)

                #if not os.path.exists(featuresFilename_2):
                f = open( featuresFilename_2, 'w' )
                f.close()


    # 3- MM ImportDataset
    #datafile-5pbr2e6nk20.c0.d3.C23.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.0.b600-_NumberOfFeatures_LabelMap.txt
    #datafile-5pbr2e6nk20.c0.d3.C23.N401.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.0.b600-train_MultiLabeled.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            #if fnmatch.fnmatch(myfile, '*_NumberOfFeatures_LabelMap.txt'):
            if fnmatch.fnmatch(myfile, '*train.arff'):
                # remove tmp files first created by previous processes
                removeTempFiles()

                numFeaturesFile = myfile[:-10] + "_NumberOfFeatures_LabelMap.txt"
                numFeaturesFile = os.path.join(FIELS_PATH, numFeaturesFile)

                fileLines = [line.strip() for line in open(numFeaturesFile)]

                for fileLine in fileLines:
                    if fileLine.startswith("X_len"):
                        X_len = fileLine.split(":")[1]

                    if fileLine.startswith("Y_len"):
                        Y_len = fileLine.split(":")[1]

                trainingMultiLabelFile = myfile[:-10] + "train_MultiLabeled.arff"
                trainingMultiLabelFile = os.path.join(FIELS_PATH, trainingMultiLabelFile)

                testingMultiLabelFile = myfile[:-10] + "test_MultiLabeled.arff"
                testingMultiLabelFile = os.path.join(FIELS_PATH, testingMultiLabelFile)

                myArgs = ["java -cp ", mm_code_path+"ImportData/ ImportDataset", # space between path and class
                    "-input", str(trainingMultiLabelFile),
                    "-output", str(trainingMultiLabelFile[:-5] + "-X-train.straight"),
                    "-selectrange", "0:"+str(int(X_len)-1)
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)
                pp.wait()
                myArgs = ["java -cp ", mm_code_path+"ImportData/ ImportDataset",
                    "-input", str(trainingMultiLabelFile),
                    "-output", str(trainingMultiLabelFile[:-5] + "-Y-train.straight"),
                    "-selectrange", str(int(X_len))+":"+str(int(X_len)+int(Y_len)-1)
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)
                pp.wait()
                myArgs = ["java -cp ", mm_code_path+"ImportData/ ImportDataset", # space between path and class
                    "-input", str(testingMultiLabelFile),
                    "-output", str(testingMultiLabelFile[:-5] + "-X-test.straight"),
                    "-selectrange", "0:"+str(int(X_len)-1)
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)
                pp.wait()
                myArgs = ["java -cp ", mm_code_path+"ImportData/ ImportDataset",
                    "-input", str(testingMultiLabelFile),
                    "-output", str(testingMultiLabelFile[:-5] + "-Y-test.straight"),
                    "-selectrange", str(int(X_len))+":"+str(int(X_len)+int(Y_len)-1)
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)
                pp.wait()
    # 4- MM TrainTest
                XtrainStraightFile = myfile[:-10] + "train_MultiLabeled-X-train.straight"
                XtrainStraightFile = os.path.join(FIELS_PATH, XtrainStraightFile)


                YtrainStraightFile = myfile[:-10] + "train_MultiLabeled-Y-train.straight"
                YtrainStraightFile = os.path.join(FIELS_PATH, YtrainStraightFile)

                XtestStraightFile = myfile[:-10] + "test_MultiLabeled-X-test.straight"
                XtestStraightFile = os.path.join(FIELS_PATH, XtestStraightFile)

                YtestStraightFile = myfile[:-10] + "test_MultiLabeled-Y-test.straight"
                YtestStraightFile = os.path.join(FIELS_PATH, YtestStraightFile)

                myArgs = ["java -cp ", mm_code_path+"vldb15/ TrainTest", # space between path and class
                    "-alg",  "opt",
                    "-X_train", str(XtrainStraightFile),
                    "-Y_train", str(YtrainStraightFile),
                    "-X_test", str(XtestStraightFile),
                    "-Y_test", str(YtestStraightFile),
                    "-k_min", "2",
                    "-k_max", str(X_len),
                    "-dk", "2",
                    "-stats", str(numFeaturesFile[:-30] + "train-test.txt"),
                    "-show", "yes"
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pp.wait()
                #print pp.stdout
                trainTestStdout = []
                currK = 2
                bestK = 2
                maxK = 2
                minErr = float(999999.0)

                for line in pp.stderr: # gives the exception (k value=318is bigger than max_k=317)
                    print line
                    #k=318: k value=318is bigger than max_k=317
                    lineList_maxK = []
                    lineList_maxK = line.split(" ")
                    #print "line"
                    #print line
                    #print "lineList_maxK"
                    #print lineList_maxK
                    if lineList_maxK.__contains__("bigger"):
                        #print "got here"
                        maxK = int(lineList_maxK[4].split("=")[1])

                for line in pp.stdout:
                    line = line.rstrip()
                    trainTestStdout.append(line.rstrip())

                for line in trainTestStdout:
                    line = line.rstrip()
                    ##k=318: k value=318is bigger than max_k=317
                    #lineList_maxK = []
                    #lineList_maxK = line.split(" ")
                    ##print "line"
                    ##print line
                    ##print "lineList_maxK"
                    ##print lineList_maxK
                    #if lineList_maxK.__contains__("bigger"):
                    #    #print "got here"
                    #    maxK = int(lineList_maxK[5].split("=")[1])

                    #print len(lineList_maxK)
                    #print lineList_maxK[0]
                    #print lineList_maxK[1]
                    #if len(lineList_maxK) > 0:
                    #    maxK = int(lineList_maxK[1].split("=")[1])

                    #k=290: regression err=0.566614 classification err=0.280625
                    if len(line) > 0 and line[0] == 'k':
                        lineList_bestK = line.split(" ")
                        currK = int(lineList_bestK[0].split(":")[0].split("=")[1])
                        #print lineList_bestK
                        #if not lineList_bestK.__contains__("bigger"): #to avoid k=318: k value=318is bigger than max_k=317
                        if len(lineList_bestK) == 5:
                            currErr = float(lineList_bestK[4].split("=")[1])
                            #print "currErr: " + str(currErr)
                            if currErr < minErr:
                                bestK = currK
                                minErr = currErr

                #print maxK
                #print bestK

                myArgs = ["java -cp ", mm_code_path+"vldb15/ TrainTest", # space between path and class
                    "-alg",  "opt",
                    "-X_train", str(XtrainStraightFile),
                    "-Y_train", str(YtrainStraightFile),
                    "-X_test", str(XtestStraightFile),
                    "-Y_test", str(YtestStraightFile),
                    "-k_min", str(bestK),
                    "-k_max", str(bestK),
                    "-dk", "1",
                    "-stats", str(numFeaturesFile[:-30] + "train-test.txt"),
                    "-show", "yes"
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pp.wait()
                for line in pp.stdout: # gives the exception (k value=318is bigger than max_k=317)
                    print line
                    if len(line) > 0:
                        if line[0] == "{":
                            f = open( numFeaturesFile[:-30] + "features.arff", 'w' )
                            f.write(line)
                            f.close()

                myArgs = ["java -cp ", mm_code_path+"vldb15/ TrainTest", # space between path and class
                    "-alg",  "opt",
                    "-X_train", str(XtrainStraightFile),
                    "-Y_train", str(YtrainStraightFile),
                    "-X_test", str(XtestStraightFile),
                    "-Y_test", str(YtestStraightFile),
                    "-k_min", str(maxK),
                    "-k_max", str(maxK),
                    "-dk", "1",
                    "-stats", str(numFeaturesFile[:-30] + "train-test.txt"),
                    "-show", "yes"
                    ]

                print ' '.join(myArgs)

                pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pp.wait()
                for line in pp.stdout: # gives the exception (k value=318is bigger than max_k=317)
                    print line
                    if len(line) > 0:
                        if line[0] == "{":
                            f = open( numFeaturesFile[:-30] + "features_2.arff", 'w' )
                            f.write(line)
                            f.close()

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
            #newTrainList.append(line) # no need as MM algo complaines about format (@ATTRIBUTE lines added)
            continue
        else:
            webpage = line.split(",")[-1]
            instanceWithoutLabel = ",".join(line.split(",")[:-1])
            newTrainList.append(instanceWithoutLabel + "," + ",".join(str(i) for i in labelMap[webpage]))

    noFearures = len(line.split(",")[:-1]) # excluding class name

    for line in testList:
        if line[0] == '@':
            #newTestList.append(line)  # no need as MM algo complaines about format @ lines added
            continue
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

    fLabelMap.write('X_len, Number of features:' + str(noFearures)+'\n')
    fLabelMap.write('Y_len, Number of lables:' + str(len(labelMap.values()[0]))+'\n\n\n')


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

def  removeTempFiles():
    #FIELS_PATH2 = FIELS_PATH.split("cache")[0]
    for (path, dirs, files) in os.walk("."):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*.tmp'):
                toBeDelFile = os.path.join(".", myfile)
                if os.path.exists(toBeDelFile):
                    print toBeDelFile + " to be deleted!"
                    os.remove(toBeDelFile)

if __name__ == "__main__":
    main()