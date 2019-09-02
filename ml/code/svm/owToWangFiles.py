import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch
import sys

# Open World: generating Wang files
def main():
    '''


    trainingFilename = "datafile-openworld100.jb3tewaok1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-train"
    testingFilename  = "datafile-openworld100.jb3tewaok1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-test"
    outputFilename   = "k1100.c0.d0.C16.N2000.t90.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0"
    config.NUM_MONITORED_SITES = 100

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld5.1ecd5e2e-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 5


    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-
    .034bed41-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
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

    trainingFilename = "trainingEntity_NaiveBayes_80_datafile-openworld80.ac6b2ce0-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_80_datafile-openworld80.ac6b2ce0-.k100.c0.d0.C17.N100.t16.T4.NoAcktrue.w"
    outputFilename   = "openworld80.ac6b2ce0-.k100.c0.d0.C17.N100.t16.T4"
    config.NUM_MONITORED_SITES = 80
    '''
    '''
    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld5.6d618df0-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld5.6d618df0-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld5.6d618df0-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 5


    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld10.1e010ceb-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld10.1e010ceb-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld10.1e010ceb-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 10

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld15.0705642a-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld15.0705642a-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld15.0705642a-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 15

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld20.6bf8921f-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld20.6bf8921f-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld20.6bf8921f-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 20

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld40.5ffe2b59-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld40.5ffe2b59-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld40.5ffe2b59-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 40

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld60.60b4f6a5-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld60.60b4f6a5-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld60.60b4f6a5-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 60

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-openworld80.0be88f1b-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-openworld80.0be88f1b-.k100.c0.d3.C17.N101.t16.T4.NoAcktrue.w"
    outputFilename   = "Android_openworld80.0be88f1b-.k100.c0.d3.C17.N101.t16.T4"
    config.NUM_MONITORED_SITES = 80
    '''

    '''
    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld5"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld5"
    outputFilename   = "HTTPS_openworld5.45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 5


    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld10"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld10"
    outputFilename   = "HTTPS_openworld10.45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 10

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld15"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld15"
    outputFilename   = "HTTPS_openworld15.45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 15

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld20"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld20"
    outputFilename   = "HTTPS_45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld20"
    config.NUM_MONITORED_SITES = 20

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld40"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld40"
    outputFilename   = "HTTPS_openworld40.45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 40

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld60"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld60"
    outputFilename   = "HTTPS_openworld60.45e6fe66-.k100.c0.d0.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 60


    '''

    '''
    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld5"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld5"
    outputFilename   = "Android_openworld5.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 5


    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld10"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld10"
    outputFilename   = "Android_openworld10.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 10

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld15"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld15"
    outputFilename   = "Android_openworld15.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 15

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld20"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld20"
    outputFilename   = "Android_openworld20.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 20

    trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld40"
    testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld40"
    outputFilename   = "Android_openworld40.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    config.NUM_MONITORED_SITES = 40
    '''
    #trainingFilename = "trainingEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld60"
    #testingFilename  = "testEntity_NaiveBayes_140_datafile-36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1.openworld60"
    #outputFilename   = "Android_openworld60.36e28fe9-.k100.c0.d3.C17.N128.t16.T4.NoAcktrue.wPC1.CPD1"
    #config.NUM_MONITORED_SITES = 60
    '''

    '''

    #trainingFilename = "datafile-openworld5.vu5ax1wyk100.c0.d0.C16.N755.t16.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-train-orig"
    #testingFilename  = "datafile-openworld5.vu5ax1wyk100.c0.d0.C16.N755.t16.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0-test"
    #outputFilename   = "openworld5.vu5ax1wyk100.c0.d0.C16.N755.t16.T4.D1.E1.F1.G1.H1.I1.B64.J8.K300.L0.05.M100.A0"
    #config.NUM_MONITORED_SITES = 5

    #__writeWang(trainingFilename, testingFilename, outputFilename)

    # setwise features to Wang open world - knn files
    #FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/website-fingerprinting-master-latestglove4f5-ACK-deploy-OpenWorld/wang/output_setwise_openworld_new_Android_k400_for_knnw/androidtor_2"
    #FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/website-fingerprinting-master-latestglove4f5-ACK-deploy-OpenWorld/wang/output_setwise_openworld_new_Android_k400_for_knnw/trainNoCPDTestCPD"
    #FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/website-fingerprinting-master-latestglove4f5-ACK-deploy-OpenWorld/wang/output_setwise_openworld_new_Android_k400_for_knnw/trainCPDTestCPD"
    '''
    FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/website-fingerprinting-master-latestglove4f5-ACK-deploy-OpenWorld/wang/output_setwise_openworld_new_Android_k100_for_knnw/NoCPD"
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, 'trainingEntity*arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = 'testEntity' + myfile[14:]
                testingFilename = os.path.join(FIELS_PATH, testingFilename)
                outputFilename = myfile[15:-17]
                outputFilename = os.path.join(FIELS_PATH, outputFilename)
                config.NUM_MONITORED_SITES = int(myfile.split("openworld")[1].split(".")[0]) # ...openworld5.aab...

                print trainingFilename
                print testingFilename
                print outputFilename
                print config.NUM_MONITORED_SITES
                print '\n'

                __writeWang(trainingFilename, testingFilename, outputFilename)


    FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-Glove-23sep2015/wang/output_BiDir_openworld_d0/NoCPD/cache"
    #datafile-openworld5.bvac19hwk300.c0.d0.C22.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V1-train.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)
                outputFilename = myfile[9:-11]
                outputFilename = os.path.join(FIELS_PATH, outputFilename)
                config.NUM_MONITORED_SITES = int(myfile.split("openworld")[1].split(".")[0]) # ...openworld5.aab...

                print trainingFilename
                print testingFilename
                print outputFilename
                print config.NUM_MONITORED_SITES
                print '\n'

                __writeWang(trainingFilename, testingFilename, outputFilename)

    '''

    #FIELS_PATH="/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-Glove-23sep2015/wang/output_BiDir_openworld_d3/NoCPD/cache-d3"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/2UsenixPaper16/OpenWorld/code/cache"
    #datafile-openworld5.15l8byg7k400.c0.d3.C22.N401.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V1-test.arff
    #datafile-openworld5.15l8byg7k400.c0.d3.C22.N401.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V1-train.arff

    if len(sys.argv) < 2:
        print "Please pass FIELS_PATH."
        sys.exit(2)

    FIELS_PATH = str(sys.argv[1])

    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)
                outputFilename = myfile[9:-11]
                outputFilename = os.path.join(FIELS_PATH, outputFilename)
                config.NUM_MONITORED_SITES = int(myfile.split("openworld")[1].split(".")[0]) # ...openworld5.aab...

                print trainingFilename
                print testingFilename
                print outputFilename
                print config.NUM_MONITORED_SITES
                #print '\n'

                __writeWang(trainingFilename, testingFilename, outputFilename)


def __writeWang(trainingFilename, testingFilename, outputFilename):
    '''
    openWorldTrainingArffFile = os.path.join(config.WANG, trainingFilename + '.arff')
    openWorldTestingArffFile = os.path.join(config.WANG, testingFilename + '.arff')
    folderName = os.path.join(config.WANG, outputFilename)
    '''

    openWorldTrainingArffFile = trainingFilename
    openWorldTestingArffFile = testingFilename
    folderName = outputFilename

    if not os.path.exists(folderName):
        os.mkdir(folderName)
    else:
        pass
        #shutil.rmtree(folderName) # delete and remake folder
        #os.mkdir(folderName)

    folderName = os.path.join(folderName, 'batch')

    if not os.path.exists(folderName):
        os.mkdir(folderName)
    else:
        print "batch folder already exisits! return.\n\n"
        return
        #shutil.rmtree(folderName) # delete and remake folder
        #os.mkdir(folderName)


    trainList = Utils.readFile1(openWorldTrainingArffFile)
    testList = Utils.readFile1(openWorldTestingArffFile)

    currentWebsite = ""
    fileCtr = 0


    for line in trainList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
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

    # adding the instances from the testing arff file for the monitored classes
    testingFileInitialCtr = fileCtr # if -t 16 and -T 4 then this counter will be 16 as the created files from train.arff are 0-0f ... 0-15f
    currentWebsite = monitoredClasses[0]

    for line in testList:
        if line[0] == '@':
            pass
        else:
            lineArray = line.split(",")
            website = lineArray[-1]
            if monitoredClasses.__contains__(website):
                if website == currentWebsite:
                    filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + "f"
                    f = open(os.path.join(folderName, filename), 'w')
                    f.write(" ".join(lineArray[:-1] ))
                    f.close()
                    fileCtr = fileCtr + 1
                else:
                    currentWebsite = website
                    fileCtr = testingFileInitialCtr
                    filename = str(monitoredClasses.index(website)) + "-" + str(fileCtr) + "f"
                    f = open(os.path.join(folderName, filename), 'w')
                    f.write(" ".join(lineArray[:-1] ))
                    f.close()
                    fileCtr = fileCtr + 1

    # adding the instances from the testing arff file for the unmonitored classes
    for line in testList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                unMonitoredClasses = getUnMonitoredClasses(line, monitoredClasses)
                currentWebsite == unMonitoredClasses[0]
        else:
            lineArray = line.split(",")
            website = lineArray[-1]
            if unMonitoredClasses.__contains__(website): # last instance of that class will override and we will have one file only (so one instance per unmonitored)
                filename = str(unMonitoredClasses.index(website)) + "f"
                f = open(os.path.join(folderName, filename), 'w')
                f.write(" ".join(lineArray[:-1] ))
                f.close()





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


if __name__ == "__main__":
    main()
