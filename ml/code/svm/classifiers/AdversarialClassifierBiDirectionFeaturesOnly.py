# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math

import numpy as np
from sklearn.decomposition import PCA

import config
from Utils import Utils

class AdversarialClassifierBiDirectionFeaturesOnly:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def roundNumberMarker(n):
        if n==4 or n==5: return 3
        elif n==7 or n==8: return 6
        elif n==10 or n==11 or n==12 or n==13: return 9
        else: return n

    @staticmethod
    def traceToInstance( trace ):

        instance = {}

        if trace.getPacketCount()==0:
            instance = {}
            instance['class'] = 'webpage'+str(trace.getId())
            #instance['class'] = 'yes'
            '''
            if (config.DATA_SOURCE==64 or config.DATA_SOURCE==65) and trace.getId() < config.NUM_BENIGN_CLASSES:
                    instance['class'] = 'webpage0' # all benign have the same label
                    #instance['class'] = 'yes'
            else:
                instance['class'] = 'webpage'+str(trace.getId())
            '''
            return instance

        if config.GLOVE_OPTIONS['packetSize'] == 1:
            instance = trace.getHistogram()
        '''
            if config.DATA_SOURCE == 5: # WANG14 dataset
                #Unique packet lengths
                for i in range(-1500, 1501): # as in Wang's kNN paper
                    key = "0-" # uplink
                    if i < 0:
                        key = "1-" # downlink
                    key += str(i)
                    if not instance.get( key ):
                        instance[key] = 0
        '''
        # Size/Number Markers
        directionCursor = None
        dataCursor      = 0
        numberCursor    = 0
        for packet in trace.getPackets():
            #print packet.getLength()
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:

                if config.GLOVE_OPTIONS['burstSize'] == 1:
                    #dataKey = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
                    dataKey = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, config.bucket_Size) )

                    if not instance.get( dataKey ):
                        instance[dataKey] = 0
                    instance[dataKey] += 1

                if config.GLOVE_OPTIONS['burstNumber'] == 1:
                    numberKey = 'N'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundNumberMarker(numberCursor) )
                    if not instance.get( numberKey ):
                        instance[numberKey] = 0
                    instance[numberKey] += 1

                directionCursor = packet.getDirection()
                dataCursor      = 0
                numberCursor    = 0

            dataCursor += packet.getLength()
            numberCursor += 1

        if dataCursor>0:

            if config.GLOVE_OPTIONS['burstSize'] == 1:
                #key = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
                key = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, config.bucket_Size) )

                if not instance.get( key ):
                    instance[key] = 0
                instance[key] += 1

        if numberCursor>0:

            if config.GLOVE_OPTIONS['burstNumber'] == 1:
                numberKey = 'N'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundNumberMarker(numberCursor) )
                if not instance.get( numberKey ):
                    instance[numberKey] = 0
                instance[numberKey] += 1

        # HTML Markers
        counterUP = 0
        counterDOWN = 0
        htmlMarker = 0
        for packet in trace.getPackets():
            if packet.getDirection() == Packet.UP:
                counterUP += 1
                if counterUP>1 and counterDOWN>0: break
            elif packet.getDirection() == Packet.DOWN:
                counterDOWN += 1
                htmlMarker += packet.getLength()

        htmlMarker = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( htmlMarker, 600 )
        #instance['H'+str(htmlMarker)] = 1

        # Ocurring Packet Sizes
        packetsUp = []
        packetsDown = []
        for packet in trace.getPackets():
            if packet.getDirection()==Packet.UP and packet.getLength() not in packetsUp:
                packetsUp.append( packet.getLength() )
            if packet.getDirection()==Packet.DOWN and packet.getLength() not in packetsDown:
                packetsDown.append( packet.getLength() )
        #instance['uniquePacketSizesUp'] = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( len( packetsUp ), 2)
        #instance['uniquePacketSizesDown'] = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( len( packetsDown ), 2)

        # Percentage Incoming Packets
        #instance['percentageUp']   = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( 100.0 * trace.getPacketCount( Packet.UP ) / trace.getPacketCount(), 5)
        #instance['percentageDown'] = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( 100.0 * trace.getPacketCount( Packet.DOWN ) / trace.getPacketCount(), 5)

        # Number of Packets
        #instance['numberUp']   = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( trace.getPacketCount( Packet.UP ), 15)
        #instance['numberDown'] = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( trace.getPacketCount( Packet.DOWN ), 15)

        # Total Bytes Transmitted
        bandwidthUp   = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( trace.getBandwidth( Packet.UP ),   10000)
        bandwidthDown = AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary( trace.getBandwidth( Packet.DOWN ), 10000)
        #instance['0-B'+str(bandwidthUp)] = 1
        #instance['1-B'+str(bandwidthDown)] = 1


        # Adversarial Apr 29, 2015
        directionCursor = None
        dataCursor      = 0

        prevDataCursor = 0
        prevDirectionCursor = None
        secondBurstAndUp = False

        timeCursor = 0
        prevTimeCursor = 0
        burstTimeRef = 0

        timeBase = 1
        if (config.DATA_SOURCE == 5): timeBase = config.bucket_Time # works well with Wang Tor

        #sizeBase = 600
        sizeBase = config.bucket_Size

        burstTimeList = [] # uplink and downlink
        upBurstTimeList = []
        downBurstTimeList = []

        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:

                ##dataKey = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
                ##if not instance.get( dataKey ):
                ##    instance[dataKey] = 0
                ##instance[dataKey] += 1

                if config.GLOVE_OPTIONS['burstTime'] == 1:
                    timeKey = 'T'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( timeKey ):
                        instance[timeKey] = 0
                    instance[timeKey] += 1

                burstTimeList.append(timeCursor)

                if (directionCursor==0):
                    upBurstTimeList.append(timeCursor)

                if (directionCursor==1):
                    downBurstTimeList.append(timeCursor)



                #prevDirectionCursor = directionCursor

                #directionCursor = packet.getDirection()

                burstTimeRef = packet.getTime()

                # BiBurst
                if secondBurstAndUp:

                    # PairBurst (up_down) (0_1)
                    #if prevDirectionCursor==0 and directionCursor==1:
                    #if prevDirectionCursor==1 and directionCursor==0:

                    if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                        pairBurstDataKey = 'biSize-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevDataCursor, sizeBase) )+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, sizeBase) )
                        if not instance.get( pairBurstDataKey ):
                            instance[pairBurstDataKey] = 0
                        instance[pairBurstDataKey] += 1

                    # time
                    if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                        biBurstTimeKey = 'biTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                        if not instance.get( biBurstTimeKey ):
                            instance[biBurstTimeKey] = 0
                        instance[biBurstTimeKey] += 1

                    # PairBurst (up_down) (0_1)
                    '''
                    #if prevDirectionCursor==0 and directionCursor==1:
                    if prevDirectionCursor==1 and directionCursor==0:
                        pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                         str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )

                        if prevTimeCursor < 0:
                            print 'Negative time ' + str(prevTimeCursor)

                        if not instance.get( pairBurstTimeKey ):
                            instance[pairBurstTimeKey] = 0
                        instance[pairBurstTimeKey] += 1
                    '''
                prevDirectionCursor = directionCursor

                directionCursor = packet.getDirection()

                prevDataCursor = dataCursor
                dataCursor     = 0

                prevTimeCursor = timeCursor

                timeCursor = 0

                secondBurstAndUp = True

            dataCursor += packet.getLength()
            #print 'packet time ' + str(packet.getTime())
            #print 'time cursor before updating ' + str(timeCursor)
            timeCursor = packet.getTime() - burstTimeRef
            #print 'time cursor after updating ' + str(timeCursor)

        if dataCursor>0:

            #key = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
            #if not instance.get( key ):
            #    instance[key] = 0
            #instance[key] += 1

            if config.GLOVE_OPTIONS['burstTime'] == 1:
                timeKey = 'T'+str(directionCursor)+'-'+str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                if not instance.get( timeKey ):
                    instance[timeKey] = 0
                instance[timeKey] += 1

            burstTimeList.append(timeCursor)

            if (directionCursor==0):
                upBurstTimeList.append(timeCursor)

            if (directionCursor==1):
                downBurstTimeList.append(timeCursor)

            # BiBurst
            if secondBurstAndUp:

                # PairBurst (up_down) (0_1)
                #if prevDirectionCursor==0 and directionCursor==1:
                #if prevDirectionCursor==1 and directionCursor==0:
                if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                    pairBurstDataKey = 'biSize-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevDataCursor, sizeBase) )+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(dataCursor, sizeBase) )
                    if not instance.get( pairBurstDataKey ):
                        instance[pairBurstDataKey] = 0
                    instance[pairBurstDataKey] += 1

                # time
                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    biBurstTimeKey = 'biTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( biBurstTimeKey ):
                        instance[biBurstTimeKey] = 0
                    instance[biBurstTimeKey] += 1

                # PairBurst (up_down) (0_1)
                '''
                #if prevDirectionCursor==0 and directionCursor==1:
                if prevDirectionCursor==1 and directionCursor==0:
                    pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversarialClassifierBiDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( pairBurstTimeKey ):
                        instance[pairBurstTimeKey] = 0
                    instance[pairBurstTimeKey] += 1
                '''
        #instance['bandwidthUp'] = trace.getBandwidth( Packet.UP ) # % 1000
        #instance['bandwidthDown'] = trace.getBandwidth( Packet.DOWN ) # % 1000
        '''
        maxTime = 0
        for packet in trace.getPackets():
             if packet.getTime() > maxTime:
                 maxTime = packet.getTime()
        instance['time'] = maxTime # % 1000
        '''


        # Five number summary
        if config.FIVE_NUM_SUM == 1:
            if burstTimeList.__len__()>0:
                a = np.array(burstTimeList)

                instance['burstTimeMin']    = a.min()
                instance['burstTimeLowerQuartile']    = np.percentile(a, 25)
                instance['burstTimeMedian'] = np.percentile(a, 50)
                instance['burstTimeUpperQuartile']    = np.percentile(a, 75)
                instance['burstTimeMax']    = a.max()
            else:
                instance['burstTimeMin']    = 0
                instance['burstTimeLowerQuartile']    = 0
                instance['burstTimeMedian'] = 0
                instance['burstTimeUpperQuartile']    = 0
                instance['burstTimeMax']    = 0

            if upBurstTimeList.__len__()>0:
                b = np.array(upBurstTimeList)

                instance['upBurstTimeMin']    = b.min()
                instance['upBurstTimeLowerQuartile']    = np.percentile(b, 25)
                instance['upBurstTimeMedian'] = np.percentile(b, 50)
                instance['upBurstTimeUpperQuartile']    = np.percentile(b, 75)
                instance['upBurstTimeMax']    = b.max()
            else:
                instance['upBurstTimeMin']    = 0
                instance['upBurstTimeLowerQuartile']    = 0
                instance['upBurstTimeMedian'] = 0
                instance['upBurstTimeUpperQuartile']    = 0
                instance['upBurstTimeMax']    = 0

            if downBurstTimeList.__len__()>0:
                c = np.array(downBurstTimeList)

                instance['downBurstTimeMin']    = c.min()
                instance['downBurstTimeLowerQuartile']    = np.percentile(c, 25)
                instance['downBurstTimeMedian'] = np.percentile(c, 50)
                instance['downBurstTimeUpperQuartile']    = np.percentile(c, 75)
                instance['downBurstTimeMax']    = c.max()
            else:
                instance['downBurstTimeMin']    = 0
                instance['downBurstTimeLowerQuartile']    = 0
                instance['downBurstTimeMedian'] = 0
                instance['downBurstTimeUpperQuartile']    = 0
                instance['downBurstTimeMax']    = 0


        # Label

        '''
        if (config.DATA_SOURCE==64 or config.DATA_SOURCE==65) and trace.getId() < config.NUM_BENIGN_CLASSES:
                instance['class'] = 'webpage0' # all benign have the same label
                #instance['class'] = 'yes'
        else:
            instance['class'] = 'webpage'+str(trace.getId())
            #instance['class'] = 'yes'
        '''
        instance['class'] = 'webpage'+str(trace.getId())

        return instance


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

        if config.OC_SVM == 0: # multi-class svm
            if config.CROSS_VALIDATION == 0:
                return wekaAPI.execute( trainingFile,
                                 testingFile,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072', # Cost
                                  #'-S','2', # one-class svm
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
                                  '-B'] ) # confidence
        else: # one-class svm
            if config.CROSS_VALIDATION == 0:
                print str(config.SVM_KERNEL)
                print str(config.OC_SVM_Nu)
                return wekaAPI.executeOneClassSVM( trainingFile,
                                 testingFile,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K',str(config.SVM_KERNEL),
                                  #'-K','2', # RBF kernel
                                  #'-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  #'-C','131072', # Cost
                                  #'-N','0.001', # nu
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
    #one class svm
    if config.CROSS_VALIDATION == 0:
        return wekaAPI.executeOneClassSVM( trainingFile,
                         testingFile,
                         "weka.Run weka.classifiers.functions.LibSVM",
                         ['-K','2', # RBF kernel
                          '-G','0.0000019073486328125', # Gamma
                          ##May20 '-Z', # normalization 18 May 2015
                          '-C','131072', # Cost
                          #'-N','0.2', # nu, def: 0.5
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


















    @staticmethod
    def classify(runID, trainingSet, testingSet):
        print 'DT'
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile,
                               testingFile,
                               "weka.classifiers.trees.J48",
                               ['-C', '0.25',
                                '-M', '2'])

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA4([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             [#'-K','0', # Linear kernel
                              '-K','2', # RBF kernel
                              #'-G','0.0000019073486328125', # Gamma
                              '-G','0.000030518',
                              ##May20 '-Z', # normalization 18 May 2015
                              #'-C','131072',
                              '-C','8'] ) # Cost



    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA6([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072'] ) # Cost


    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )


    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.classifiers.trees.RandomForest",
                             ['-I','10', #
                              '-K','0', #
                              '-S','1'] ) #

    '''