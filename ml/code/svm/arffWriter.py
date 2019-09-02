# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders


def writeArffFiles( runID, trainingSet, testingSet ):

    # For Open World
    outputFilenameArray = [
                       'k' + str(config.BUCKET_SIZE),
                       'c' + str(config.COUNTERMEASURE),
                       'd' + str(config.DATA_SOURCE),
                       'C' + str(config.CLASSIFIER),
                       'N' + str(config.TOP_N),
                       't' + str(config.NUM_TRAINING_TRACES),
                       'T' + str(config.NUM_TESTING_TRACES),
                       'D' + str(config.GLOVE_OPTIONS['packetSize']),
                       'E' + str(config.GLOVE_OPTIONS['burstSize']),
                       'F' + str(config.GLOVE_OPTIONS['burstTime']),
                       'G' + str(config.GLOVE_OPTIONS['burstNumber']),
                       'H' + str(config.GLOVE_OPTIONS['biBurstSize']),
                       'I' + str(config.GLOVE_OPTIONS['biBurstTime']),
                       'B' + str(config.GLOVE_OPTIONS['ModelTraceNum']),
                       'J' + str(config.GLOVE_PARAMETERS['window']),
                       'K' + str(config.GLOVE_PARAMETERS['no_components']),
                       'L' + str(config.GLOVE_PARAMETERS['learning_rate']),
                       'M' + str(config.GLOVE_PARAMETERS['epochs']),
                       'A' + str(int(config.IGNORE_ACK)),
                       'V' + str(int(config.FIVE_NUM_SUM)),
                       'P' + str(int(config.n_components_PCA)),
                       'G' + str(int(config.n_components_LDA)),
                       'l' + str(float(config.lasso)),
                       'b' + str(int(config.bucket_Size))
                       ]

    if config.COVARIATE_SHIFT != 0:
        outputFilenameArray.append('s' + str(int(config.COVARIATE_SHIFT)))

    # For Wang Tor dataset (config.DATA_SOURCE = 5) and others
    if config.NUM_NON_MONITORED_SITES != -1 and (config.DATA_SOURCE == 5 or config.DATA_SOURCE == 41 or config.DATA_SOURCE == 42 or config.DATA_SOURCE == 6 \
                                                         or config.DATA_SOURCE == 61 or config.DATA_SOURCE == 62 or config.DATA_SOURCE == 63 or config.DATA_SOURCE == 64):
        outputFilenameArray.append('u'+str(config.NUM_NON_MONITORED_SITES))

    # HP datasets
    if config.NUM_TRACE_PACKETS != -1: # num of packets to be used
        outputFilenameArray.append('p'+str(config.NUM_TRACE_PACKETS))

    # HP datasets
    if config.NUM_HP_DCOY_ATTACKS_TRAIN != -1 or config.NUM_HP_DCOY_ATTACKS_TEST != -1:
        outputFilenameArray.append('Q'+str(config.NUM_HP_DCOY_ATTACKS_TOTAL))
        outputFilenameArray.append('w'+str(config.NUM_HP_DCOY_ATTACKS_TRAIN))
        outputFilenameArray.append('W'+str(config.NUM_HP_DCOY_ATTACKS_TEST))

    outputFilename = os.path.join('.'.join(outputFilenameArray))
    runID = runID + outputFilename

    trainingFilename           = 'datafile-'+runID+'-train'
    testingFilename            = 'datafile-'+runID+'-test'



	# Change the name for open world
    if (config.NUM_MONITORED_SITES != -1):
        trainingFilename = "datafile-"+ "openworld" + str(config.NUM_MONITORED_SITES) +"." +runID+"-train-orig"
        testingFilename = "datafile-"+ "openworld" + str(config.NUM_MONITORED_SITES) +"." +runID+"-test"

    # config.NUM_NON_MONITORED_SITES is used for Wang Tor dataset only
    # need to maintain this naming convention as other scripts (like owToWangFiles.py depends on this.
    # config.DATA_SOURCE == 5 - Wang Tor dataset
    #if (config.NUM_NON_MONITORED_SITES != -1  and config.DATA_SOURCE == 5):
    #    trainingFilename = "datafile-"+ "openworld" + str(config.NUM_MONITORED_SITES) +"." +runID+"-train"
    #    testingFilename = "datafile-"+ "openworld" + str(config.NUM_MONITORED_SITES) +"." +runID+"-test"

    #print "in arff"
    classes = []
    for instance in trainingSet:
        if instance['class'] not in classes:
            classes.append(instance['class'])
    for instance in testingSet:
        if instance['class'] not in classes:
            classes.append(instance['class'])

    attributes = []
    for instance in trainingSet:
        for key in instance:
            if key not in attributes:
                attributes.append( key )
    for instance in testingSet:
        for key in instance:
            if key not in attributes:
                attributes.append( key )

    trainingFile = __writeArffFile( trainingSet, trainingFilename, classes, attributes )
    testingFile = __writeArffFile( testingSet, testingFilename, classes, attributes )

    # For open world only; to produce the open world training arff file
    if (config.NUM_MONITORED_SITES != -1): # and config.DATA_SOURCE != 5):
        __writeOpenWorldArffTrainingFile(trainingFilename, config.NUM_MONITORED_SITES )

        # Prepare Wang datasets
        # Commented on Dec 16, 2015. Uncomment if needed.
        #__writeWang(trainingFilename, testingFilename, outputFilename)

    return [trainingFile, testingFile]


def __writeArffFile( inputArray, outputFile, classes, attributes ):
    arffFile = []

    attributes = sorted(attributes) # Khaled 10/04/2015

    arffFile.append('@RELATION sites')
    for attribute in attributes:
        if attribute!='class':
            arffFile.append('@ATTRIBUTE '+str(attribute)+' real')
    arffFile.append('@ATTRIBUTE class {'+','.join(classes)+'}')
    arffFile.append('@DATA')

    for instance in inputArray:
        tmpBuf = []
        for attribute in attributes:
            if attribute!='class':
                val = '0'
                if instance.get(attribute) not in [None,0]:
                    val = str(instance[attribute])
                tmpBuf.append(val)
        tmpBuf.append(instance['class'])

        arffFile.append( ','.join(itertools.imap(str, tmpBuf)) )
    
    outputFile = os.path.join(config.CACHE_DIR, outputFile+'.arff')
    f = open( outputFile, 'w' )
    f.write( "\n".join( arffFile ) )
    f.close()

    return outputFile

def __writeOpenWorldArffTrainingFile(trainingFilename, numMonitoredWebsites):
    originalTrainingFile = config.CACHE_DIR + "/" + trainingFilename + ".arff"
    fileList = Utils.readFile(originalTrainingFile)
    newList = []
    monitoredClasses = []

    for line in fileList:
        if line[0] == '@':
            if not line.startswith("@ATTRIBUTE class"):
                newList.append(line)
            else:
                monitoredClasses = getMonitoredClasses(line, numMonitoredWebsites)
                newList.append("@ATTRIBUTE class {" + ','.join(monitoredClasses) + "}")
        else:
            instanceSplit = line.split(",")

            if monitoredClasses.__contains__(instanceSplit[len(instanceSplit) - 1]):
                newList.append(line)

    #openWorldArffFile = config.CACHE_DIR + "/" + trainingFilename.substring(0, len(trainingFilename)-5) + ".arff"
    openWorldArffFile = config.CACHE_DIR + "/" + trainingFilename[:-5] + ".arff"
    f = open( openWorldArffFile, 'w' )
    #for line in newList:
    #    f.write(line)
    f.write( "\n".join( newList ) )
    f.close()


def __writeWang(trainingFilename, testingFilename, outputFilename):
    openWorldTrainingArffFile = os.path.join(config.CACHE_DIR, trainingFilename[:-5] + '.arff')
    openWorldTestingArffFile = os.path.join(config.CACHE_DIR, testingFilename + '.arff')
    folderName = os.path.join(config.WANG, outputFilename)
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    else:
        shutil.rmtree(folderName) # delete and remake folder
        os.mkdir(folderName)

    trainList = Utils.readFile(openWorldTrainingArffFile)
    testList = Utils.readFile(openWorldTestingArffFile)

    currentWebsite = ""
    fileCtr = 0


    for line in trainList:
        if line[0] == '@':
            if line.startswith("@ATTRIBUTE class"):
                monitoredClasses = getMonitoredClasses(line, config.NUM_MONITORED_SITES)
                currentWebsite = monitoredClasses[0]
        else:
            lineArray = line.split(",")
            website = lineArray[-1]
            if website == currentWebsite:
                filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + "f"
                f = open(os.path.join(folderName, filename), 'w')
                f.write(" ".join(lineArray[:-1] ))
                f.close()
                fileCtr = fileCtr + 1
            else:
                currentWebsite = website
                fileCtr = 0
                filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + "f"
                f = open(os.path.join(folderName, filename), 'w')
                f.write(" ".join(lineArray[:-1] ))
                f.close()
                fileCtr = fileCtr + 1



    for line in testList:
        if line[0] == '@':
            if line.startswith("@ATTRIBUTE class"):
                unMonitoredClasses = getUnMonitoredClasses(line, monitoredClasses)
                currentWebsite == unMonitoredClasses[0]
        else:
            lineArray = line.split(",")
            website = lineArray[-1]
            if unMonitoredClasses.__contains__(website): # last instance of that class will override and we will have one file only
                filename = str(unMonitoredClasses.index(website)) + "f"
                f = open(os.path.join(folderName, filename), 'w')
                f.write(" ".join(lineArray[:-1] ))
                f.close()





def getMonitoredClasses( classesLine, numMonitoredWebsites):
    #@ATTRIBUTE class {webpage26,webpage36,webpage48..}
    classes = classesLine.split(" ")[2]
    classes = classes.split("{")[1]
    classes = classes.split("}")[0]
    classesList = classes.split(",")

    monitoredClasses = []

    # append the first numMonitoredWebsites from the random list
    for i in range(0,numMonitoredWebsites):
        monitoredClasses.append(classesList[i])

    return monitoredClasses

def getUnMonitoredClasses( classesLine, monitoredClasses):
    #@ATTRIBUTE class {webpage26,webpage36,webpage48..}
    classes = classesLine.split(" ")[2]
    classes = classes.split("{")[1]
    classes = classes.split("}")[0]
    classesList = classes.split(",")

    unMonitoredClasses = set(classesList) - set(monitoredClasses)

    return list(unMonitoredClasses)






















