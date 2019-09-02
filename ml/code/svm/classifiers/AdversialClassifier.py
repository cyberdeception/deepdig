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

import config
from Utils import Utils

class AdversialClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance( trace ):
        instance = {}
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
        sizeBase = 600

        burstTimeList = [] # uplink and downlink
        upBurstTimeList = []
        downBurstTimeList = []
        numberCursor    = 0

        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:

                if config.GLOVE_OPTIONS['burstSize'] == 1:
                    dataKey = 'S'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                    if not instance.get( dataKey ):
                        instance[dataKey] = 0
                    instance[dataKey] += 1

                if config.GLOVE_OPTIONS['burstTime'] == 1:
                    timeKey = 'T'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
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
                                         str( AdversialClassifier.roundArbitrary(prevDataCursor, sizeBase) )+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(dataCursor, sizeBase) )
                        if not instance.get( pairBurstDataKey ):
                            instance[pairBurstDataKey] = 0
                        instance[pairBurstDataKey] += 1

                    # time
                    if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                        biBurstTimeKey = 'biTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                        if not instance.get( biBurstTimeKey ):
                            instance[biBurstTimeKey] = 0
                        instance[biBurstTimeKey] += 1

                    # PairBurst (up_down) (0_1)
                    '''
                    #if prevDirectionCursor==0 and directionCursor==1:
                    if prevDirectionCursor==1 and directionCursor==0:
                        pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )

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
                numberCursor    = 0

                prevTimeCursor = timeCursor

                timeCursor = 0

                secondBurstAndUp = True

            dataCursor += packet.getLength()
            numberCursor += 1
            #print 'packet time ' + str(packet.getTime())
            #print 'time cursor before updating ' + str(timeCursor)
            timeCursor = packet.getTime() - burstTimeRef
            #print 'time cursor after updating ' + str(timeCursor)

        if dataCursor>0:
            if config.GLOVE_OPTIONS['burstSize'] == 1:
                key = 'S'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                if not instance.get( key ):
                    instance[key] = 0
                instance[key] += 1

            if config.GLOVE_OPTIONS['burstNumber'] == 1:
                numberKey = 'N'+str(directionCursor)+'-'+str( AdversialClassifier.roundNumberMarker(numberCursor) )
                if not instance.get( numberKey ):
                    instance[numberKey] = 0
                instance[numberKey] += 1

            if config.GLOVE_OPTIONS['burstTime'] == 1:
                timeKey = 'T'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
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
                                     str( AdversialClassifier.roundArbitrary(prevDataCursor, sizeBase) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(dataCursor, sizeBase) )
                    if not instance.get( pairBurstDataKey ):
                        instance[pairBurstDataKey] = 0
                    instance[pairBurstDataKey] += 1

                # time
                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    biBurstTimeKey = 'biTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( biBurstTimeKey ):
                        instance[biBurstTimeKey] = 0
                    instance[biBurstTimeKey] += 1

                # PairBurst (up_down) (0_1)
                '''
                #if prevDirectionCursor==0 and directionCursor==1:
                if prevDirectionCursor==1 and directionCursor==0:
                    pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( pairBurstTimeKey ):
                        instance[pairBurstTimeKey] = 0
                    instance[pairBurstTimeKey] += 1
                '''
        instance['bandwidthUp'] = trace.getBandwidth( Packet.UP ) # % 1000
        instance['bandwidthDown'] = trace.getBandwidth( Packet.DOWN ) # % 1000
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
        instance['class'] = 'webpage'+str(trace.getId())

        return instance


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

    '''

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              '-C','131072',
                              '-Z']) # normalization 18 May 2015] ) # Cost
    '''