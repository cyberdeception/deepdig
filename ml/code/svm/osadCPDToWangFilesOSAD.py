import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch

# Open World: generating Wang files
def main():
    '''
     Put the files in the cache directory

    trainingFilename = "datafile-openworld100.jb3tewaok1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-train"
    testingFilename  = "datafile-openworld100.jb3tewaok1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-test"
    outputFilename   = "k1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0"
    config.NUM_MONITORED_SITES = 100

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 5


    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld10.034bed41-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld10.034bed41-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld10.034bed41-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 10

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld15.4ba73020-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld15.4ba73020-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld15.4ba73020-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 15

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld20.384c74f1-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld20.384c74f1-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld20.384c74f1-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 20

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld40.fc0a449e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld40.fc0a449e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld40.fc0a449e-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 40

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld60.3a11cefb-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld60.3a11cefb-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld60.3a11cefb-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 60

    trainingFilename = "datafile-fa847a60-.k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w-train"
    testingFilename  = "datafile-fa847a60-.k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w.PC1.CPD1-test"
    outputFilename   = "k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w.PC1.CPD1"
    config.NUM_MONITORED_SITES = 100
    '''

    #trainingFilename = "datafile-ca824ca5-.k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w-train"
    #testingFilename  = "datafile-ca824ca5-.k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w.PC1.CPD1-test"
    #outputFilename   = "k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w.PC1.CPD1"

    #trainingFilename = "datafile-fa847a60-.k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w-train(copy)"
    #testingFilename  = "datafile-fa847a60-.k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w.PC1.CPD1-test(copy)"
    #outputFilename   = "k100.c0.d0.C19.N775.t10.T30.NoAcktrue.w.PC1.CPD1(copy)"

    # HTTPS
    #trainingFilename = "datafile-ee123ffd-.k150.c0.d0.C19.N775.t30.T30.NoAcktrue.w-train"
    #testingFilename  = "datafile-ee123ffd-.k150.c0.d0.C19.N775.t30.T30.NoAcktrue.w.PC1.CPD1-test"
    #outputFilename   = "ee123ffd-.k150.c0.d0.C19.N775.t30.T30.NoAcktrue.w.PC1.CPD1"

    # Android Device
    trainingFilename = "datafile-a9b1b7ac-.k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w-train"
    testingFilename  = "datafile-a9b1b7ac-.k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w.PC1.CPD1-test"
    outputFilename   = "a9b1b7ac-.k150.c0.d3.C19.N400.t30.T30.NoAcktrue.w.PC1.CPD1"

    __writeWang(trainingFilename, testingFilename, outputFilename)


# Testing File should be without '@' headers (just instances)
def __writeWang(trainingFilename, testingFilename, outputFilename):
    closedWorldTrainingArffFile = os.path.join(config.WANG_OSAD, trainingFilename + '.arff')
    closedWorldTestingArffFile = os.path.join(config.WANG_OSAD, testingFilename + '.csv')
    folderName = os.path.join(config.WANG_OSAD, outputFilename)
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    else:
        shutil.rmtree(folderName) # delete and remake folder
        os.mkdir(folderName)

    numInstancesNeeded = 20
    trainTestList = Utils.readFile3(closedWorldTrainingArffFile, closedWorldTestingArffFile, numInstancesNeeded) # numInstancesNeeded = 20 instances/class from training and 20 instances/class for testing (CPD)

#    currentWebsite = ""
#    currentEntity = 1 # first entity = 1 in the training file which is read first
    fileCtr = 0
    lengthsList = []

    #print trainTestList

    # get curr website and curr entity
    for line in trainTestList:
        if not line[0] == '@':
            lineArray = line.split(",")
            currentWebsite = lineArray[-1]
            currentEntity = lineArray[1]
            break


    for line in trainTestList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                #classes = getAllClasses(line)
                classes = getSelectedClasses(trainTestList)
                #currentWebsite = classes[0]
        else:
            #print line
            lineArray = line.split(",")
            website = lineArray[-1]
            entity = lineArray[1]
            if entity == currentEntity:
                if int(lineArray[0]) == 0:
                    lengthsList.append(lineArray[2]) # uplink
                else:
                    lengthsList.append("-"+lineArray[2]) # downlink
            else:
                #filename = str(classes.index(currentWebsite)) + "-" + str(fileCtr) + currentWebsite + " " + currentEntity + ".txt"
                filename = str(classes.index(currentWebsite) + 1) + "_" + str(fileCtr + 1) + ".txt"
                f = open(os.path.join(folderName, filename), 'w')
                f.write("\n".join(lengthsList ))
                f.close()
                fileCtr = fileCtr + 1

                lengthsList = []
                if int(lineArray[0]) == 0:
                    lengthsList.append(lineArray[2]) # uplink
                else:
                    lengthsList.append("-"+lineArray[2]) # downlink

                currentEntity = entity
                if website != currentWebsite:
                    print fileCtr
                    if fileCtr == 7:
                        print currentWebsite + " " + str(classes.index(currentWebsite))
                    fileCtr = 0
                    currentWebsite = website
                '''
                filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + "f"
                f = open(os.path.join(folderName, filename), 'w')
                f.write(" ".join(lineArray[:-1] ))
                f.close()
                fileCtr = fileCtr + 1
                '''

    # last entity
    #filename = str(classes.index(currentWebsite)) + "-" + str(fileCtr) + currentWebsite + " " + currentEntity +".txt"
    filename = str(classes.index(currentWebsite) + 1) + "_" + str(fileCtr + 1) + ".txt"
    f = open(os.path.join(folderName, filename), 'w')
    f.write("\n".join(lengthsList ))
    f.close()


    # splitting training and testing

    folderNameTrain = os.path.join(config.WANG_OSAD, outputFilename + '.Train')
    if not os.path.exists(folderNameTrain):
        os.mkdir(folderNameTrain)
    else:
        shutil.rmtree(folderNameTrain) # delete and remake folder
        os.mkdir(folderNameTrain)

    folderNameTest = os.path.join(config.WANG_OSAD, outputFilename + '.TestCPD')
    if not os.path.exists(folderNameTest):
        os.mkdir(folderNameTest)
    else:
        shutil.rmtree(folderNameTest) # delete and remake folder
        os.mkdir(folderNameTest)



    for websiteId in range(1, 101):
        #ctrTrain = 1
        #ctrTest = 1
        for (path, dirs, files) in os.walk(folderName):
            for myfile in files:
                fileNameArray = myfile.split("_") # myfile = 1_1.txt
                instanceNumber = int(fileNameArray[1].split(".")[0])
                if int(fileNameArray[0]) == websiteId:
                    if int(instanceNumber) <= numInstancesNeeded:
                        # First half is for training (No CPD) example: 1_1.txt to 1_20.txt are for training
                        filename = str(websiteId) + "_" + str(instanceNumber) + ".txt"
                        copyFiles(os.path.join(folderName, myfile), os.path.join(folderNameTrain, filename))
                    else:
                        # Second half is for testing (CPD) example: 1_21.txt to 1_40.txt are for training
                        if instanceNumber % numInstancesNeeded != 0:
                            filename = str(websiteId) + "_" + str(instanceNumber % numInstancesNeeded) + ".txt"
                        else:
                            # as instanceNumber % numInstancesNeeded = 0 in this case
                            filename = str(websiteId) + "_" + str(instanceNumber/2) + ".txt"

                        copyFiles(os.path.join(folderName, myfile), os.path.join(folderNameTest, filename))


    '''
    for line in testList:
        lineArray = line.split(",")
        website = lineArray[-1]
        entity = lineArray[1]
        if entity == currentEntity:
            if lineArray[0] == 0:
                lengthsList.append(lineArray[2]) # uplink
            else:
                lengthsList.append("-"+lineArray[2]) # downlink


        else:
            filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + ".txt"
            f = open(os.path.join(folderName, filename), 'w')
            f.write("\n".join(lengthsList ))
            f.close()
            fileCtr = fileCtr + 1

            lengthsList = []
            if lineArray[0] == 0:
                lengthsList.append(lineArray[2]) # uplink
            else:
                lengthsList.append("-"+lineArray[2]) # downlink

            currentEntity = entity
            if website != currentWebsite:
                fileCtr = 0
                currentWebsite = website


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



        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
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
    '''





def getMonitoredClasses( classesLine, numMonitoredWebsites):
    #@ATTRIBUTE class {webpage26,webpage36,webpage48..}
    classes = classesLine.split(" ")[2]
    classes = classes.split("{")[1]
    classes = classes.split("}")[0]
    classes = classes.replace(' ', '')
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

def getAllClasses( classesLine):
    #@ATTRIBUTE class {webpage26,webpage36,webpage48..}
    classes = classesLine.split(" ")[2]
    classes = classes.split("{")[1]
    classes = classes.split("}")[0]
    classes = classes.replace(' ', '')
    classesList = classes.split(",")
    '''
    monitoredClasses = []

    # append the first numMonitoredWebsites from the random list
    for i in range(0,numMonitoredWebsites):
        monitoredClasses.append(classesList[i])
    '''
    return classesList


def getSelectedClasses(trainTestList):
    classesList = []
    for line in trainTestList:

        if not (line.startswith("@") or line.startswith("Direction")):

            #print line
            lineArray = line.split(",")
            website = lineArray[-1]
            if not classesList.__contains__(website):
                classesList.append(website)


    return classesList

def copyFiles(fromFile, toFile):
    fromFileLines = [line.strip() for line in open(fromFile)]

    f = open(toFile, 'w')
    f.write("\n".join(fromFileLines))
    f.close()

if __name__ == "__main__":
    main()
