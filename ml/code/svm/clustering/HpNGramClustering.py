
import wekaAPI
import clusteringAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math

import numpy as np
from sklearn.decomposition import PCA

import config
from Utils import Utils
from EventTrace import EventTrace



class HpNGramClustering:

    @staticmethod
    def traceToInstance( eventTrace ):

        instance = {}

        if eventTrace.getEventCount()==0:
            instance = {}
            instance['class'] = 'webpage'+str(eventTrace.getId())
            return instance

        #print 'webpage'+str(eventTrace.getId())

        numMostFreqFeatures = 50
        oneGramHistogram = dict(eventTrace.getNGramHistogram(N=1, sortReverseByValue=True)) # [:numMostFreqFeatures] )

        twoGramHistogram = dict(eventTrace.getNGramHistogram(N=2, sortReverseByValue=True)) # [:numMostFreqFeatures] )
        threeGramHistogram = dict(eventTrace.getNGramHistogram(N=3, sortReverseByValue=True)) # [:numMostFreqFeatures] )
        fourGramHistogram = dict(eventTrace.getNGramHistogram(N=4, sortReverseByValue=True)) # [:numMostFreqFeatures] )
        #fiveGramHistogram = dict(eventTrace.getNGramHistogram(N=5, sortReverseByValue=True)) # [:numMostFreqFeatures] )
        #sixGramHistogram = dict(eventTrace.getNGramHistogram(N=6, sortReverseByValue=True)) # [:numMostFreqFeatures] )
        #sevenGramHistogram = dict(eventTrace.getNGramHistogram(N=7, sortReverseByValue=True)) # [:numMostFreqFeatures] )

        instance.update(oneGramHistogram)
        instance.update(twoGramHistogram)
        instance.update(threeGramHistogram)
        instance.update(fourGramHistogram)
        #instance.update(fiveGramHistogram)
        #instance.update(sixGramHistogram)
        #instance.update(sevenGramHistogram)

        # label
        instance['class'] = 'webpage'+str(eventTrace.getId())

        return instance




    @staticmethod
    def classify( runID, trainingSet, testingSet ):

        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if (config.CLUSTERING_METHOD==1):
            clusteringAPI.calcKmeans([trainingFile,testingFile],"Description goes here!")
        elif (config.CLUSTERING_METHOD==2):
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])
            clusteringAPI.calcKmeans([trainingFile,testingFile],"Description goes here!")
        elif (config.CLUSTERING_METHOD==3):
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])
            clusteringAPI.calcKmeansCvxHullDelaunay([trainingFile,testingFile],"Description goes here!")
        elif (config.CLUSTERING_METHOD==4):
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])
            clusteringAPI.calcKmeansCvxHullDelaunay_Mixed([trainingFile,testingFile],"Description goes here!")
        elif (config.CLUSTERING_METHOD==5):
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])
            clusteringAPI.calcKmeansCvxHullDelaunay_Mixed_KNN([trainingFile,testingFile],"Description goes here!", threshold=3)


        return ['NA', []]

'''
    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if config.NUM_MONITORED_SITES != -1: #no need to classify as this is for generating openworld datasets. See the line above (arffWriter)
            [accuracy,debugInfo] = ['NA', []]
            return [accuracy,debugInfo]

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA4([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        if config.lasso != 0:
            #[trainingFile,testingFile] = Utils.calcLasso3([trainingFile,testingFile])
            #[trainingFile,testingFile] = Utils.calcLogisticRegression([trainingFile,testingFile])
            Utils.calcLogisticRegression([trainingFile,testingFile])

        #Utils.plotDensity([trainingFile,testingFile])
        #Utils.plot([trainingFile,testingFile])

        if config.NUM_FEATURES_RF != 0:
            [trainingFile,testingFile] = Utils.calcTreeBaseRF([trainingFile,testingFile], config.NUM_FEATURES_RF)

        if config.OC_SVM == 0: # multi-class svm
            if config.CROSS_VALIDATION == 0:
                return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072', # Cost
                              '-B'] )  # confidence
            else:
                file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
                return wekaAPI.executeCrossValidation( file,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-x',str(config.CROSS_VALIDATION), # number of folds
                              '-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072', # Cost
                              '-B'] )  # confidence
        else: # one-class svm
            if config.CROSS_VALIDATION == 0:
                print str(config.SVM_KERNEL)
                print str(config.OC_SVM_Nu)
                return wekaAPI.executeOneClassSVM( trainingFile,
                                 testingFile,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K',str(config.SVM_KERNEL),
                                  #'-K','0', # kernel
                                  #'-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  #'-C','131072', # Cost
                                  #'-N','0.01', # nu
                                  '-N',str(config.OC_SVM_Nu), # nu
                                  '-S','2'])#, # one-class svm
                                  #'-B'] )  # confidence
            else:
                file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
                return wekaAPI.executeCrossValidation( file,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-x',str(config.CROSS_VALIDATION), # number of folds
                                  '-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072', # Cost
                                  '-B'] ) # confidence

'''