# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import re
import subprocess
import config
import os

def execute( trainingFile, testingFile, classifier, args ):
    myArgs = ["java",
        "-Xmx" + str(config.JVM_MEMORY_SIZE),
        "-classpath", '$CLASSPATH:'+config.WEKA_JAR,
        classifier,
        "-t", trainingFile,
        "-T", testingFile,
        '-v',
        '-classifications','weka.classifiers.evaluation.output.prediction.CSV'
        ]

    for arg in args:
        myArgs.append( arg )
    print ' '.join(myArgs)
    pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)

    totalPredictions = 0
    totalCorrectPredictions = 0
    debugInfo = []
    parsing = False
    for line in pp.stdout:

        line = line.rstrip()
        #print line

        if parsing == True:
            if line=='': break;
            #print line
            lineBits = line.split(',')
            actualClass = lineBits[1].split(':')[1]
            predictedClass = lineBits[2].split(':')[1]
            probEstimate = lineBits[4]
            #debugInfo.append([actualClass,predictedClass])
            debugInfo.append([actualClass,predictedClass,probEstimate])
            totalPredictions += 1.0
            if actualClass == predictedClass:
                totalCorrectPredictions += 1.0

        if line == 'inst#,actual,predicted,error,prediction':
            parsing = True

    accuracy = totalCorrectPredictions / totalPredictions * 100.0

    return [accuracy,debugInfo]


def executeCrossValidation( file, classifier, args ):
    myArgs = ["java",
        "-Xmx" + str(config.JVM_MEMORY_SIZE),
        "-classpath", '$CLASSPATH:'+config.WEKA_JAR,
        classifier,
        "-t", file,
        '-v',
        '-classifications','weka.classifiers.evaluation.output.prediction.CSV'
        ]

    for arg in args:
        myArgs.append( arg )

    #print ' '.join(myArgs)

    pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)

    totalPredictions = 0
    totalCorrectPredictions = 0
    debugInfo = []
    parsing = False
    for line in pp.stdout:
        line = line.rstrip()

        if parsing == True:
            if line=='': break;
            lineBits = line.split(',')
            actualClass = lineBits[1].split(':')[1]
            predictedClass = lineBits[2].split(':')[1]
            probEstimate = lineBits[4]
            #debugInfo.append([actualClass,predictedClass])
            debugInfo.append([actualClass,predictedClass,probEstimate])
            totalPredictions += 1.0
            if actualClass == predictedClass:
                totalCorrectPredictions += 1.0

        if line == 'inst#,actual,predicted,error,prediction':
            parsing = True

    accuracy = totalCorrectPredictions / totalPredictions * 100.0

    return [accuracy,debugInfo]


def executeOneClassSVM( trainingFile, testingFile, classifier, args ):
    myArgs = ["java",
        "-Xmx" + str(config.JVM_MEMORY_SIZE),
        "-classpath", '$CLASSPATH:'+config.WEKA_JAR,
        classifier,
        "-t", trainingFile,
        "-T", testingFile,
        '-v',
        '-classifications','weka.classifiers.evaluation.output.prediction.CSV'
        ]

    # testY = labels for testing instances, in order
    # rearrange arff, @ATTRIBUTE class {in}, for both tr and te files
    yte = rearrangeArffForOneClassSVM([trainingFile, testingFile])



    for arg in args:
        myArgs.append( arg )
    print ' '.join(myArgs)
    pp = subprocess.Popen(' '.join(myArgs), shell=True, stdout=subprocess.PIPE)

    totalPredictions = 0
    totalCorrectPredictions = 0
    debugInfo = []
    parsing = False
    yteIndx=0
    for line in pp.stdout:
        '''
        1,1:in,1:in,,1
        2,1:in,1:in,,1
        3,1:in,?,,?
        4,1:in,?,,?
        '''

        line = line.rstrip()
        print line

        if parsing == True:
            if line=='': break;
            lineBits = line.split(',')
            actualClass = yte[yteIndx]
            predictedClass = lineBits[2]#.split(':')[1]

            if predictedClass != '?':
                predictedClass = 'webpage0' # in (example of benign)
            else:
                predictedClass = 'webpage20' # example class label of out (example of attack)
            #probEstimate = lineBits[4]
            #debugInfo.append([actualClass,predictedClass])
            debugInfo.append([actualClass,predictedClass])
            totalPredictions += 1.0
            if actualClass == predictedClass: # doesn't give correct accuracy as we use webpages 0 and 20, tpr and fpr are calculated in the main program
                totalCorrectPredictions += 1.0
            yteIndx += 1

        if line == 'inst#,actual,predicted,error,prediction':
            parsing = True

    accuracy = totalCorrectPredictions / totalPredictions * 100.0 # doesn't give correct accuracy as we use webpages 0 and 20, tpr and fpr are calculated in the main program

    return [accuracy,debugInfo]


def rearrangeArffForOneClassSVM(files):
    trainList = readFile(files[0])
    testList = readFile(files[1])

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
    ###
    newTrainList = []
    newTestList = []

    newTrainList.append('@RELATION traces')
    newTestList.append('@RELATION traces')

    # selected features
    for i in range(1,len(featuresBlockArff)-2): # excluding @ATTRIBUTE class and @DATA
            newTrainList.append(featuresBlockArff[i])
            newTestList.append(featuresBlockArff[i])

    newTrainList.append('@ATTRIBUTE class {in}')
    newTestList.append( '@ATTRIBUTE class {in}')

    newTrainList.append('@DATA')
    newTestList.append('@DATA')

    for i in xrange(len(Xtr)):
        instance = Xtr[i]
        newTrainList.append(','.join([str("%.2f" % k) for k in instance]) + ',in') # all tr and te inst are labeled 'in'

    for i in xrange(len(Xte)):
        instance = Xte[i]
        newTestList.append(','.join([str("%.2f" % k) for k in instance]) + ',in') # all tr and te inst are labeled 'in'


    # writing the new training file
    fnewTrain = open(files[0], 'w')
    for item in newTrainList:
        fnewTrain.write(item+'\n')

    fnewTrain.close()

    # writing the new testing file
    fnewTest = open(files[1], 'w')
    for item in newTestList:
        fnewTest.write(item+'\n')

    fnewTest.close()

    # return actual labels for testing inst
    return yte


def readFile(fileName):
    fileLines = [line.strip() for line in open(fileName)]
    fileList = []
    for fileLine in fileLines:
        fileList.append(fileLine)

    return fileList