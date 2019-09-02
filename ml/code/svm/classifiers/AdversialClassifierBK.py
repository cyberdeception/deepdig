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

class AdversialClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance( trace ):
        instance = {}

        # Size/Number Markers
        directionCursor = None
        dataCursor      = 0

        prevDataCursor = 0
        prevDirectionCursor = None
        secondBurstAndUp = False

        timeCursor = 0
        prevTimeCursor = 0
        burstTimeRef = 0

        timeBase = 2
        sizeBase = 600
        burstTimeList = []
        upBurstTimeList = []
        downBurstTimeList = []

        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                dataKey = 'S'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                if not instance.get( dataKey ):
                    instance[dataKey] = 0
                instance[dataKey] += 1


                timeKey = 'T'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                if not instance.get( timeKey ):
                    instance[timeKey] = 0
                instance[timeKey] += 1

                burstTimeList.append(timeCursor)

                if (directionCursor==0):
                    upBurstTimeList.append(timeCursor)

                if (directionCursor==1):
                    #if timeCursor <> 0: # to be removed Jun 1, 15
                    downBurstTimeList.append(timeCursor)



                #prevDirectionCursor = directionCursor

                #directionCursor = packet.getDirection()

                burstTimeRef = packet.getTime()

                # BiBurst
                if secondBurstAndUp:
                    ##biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                    ##                 str( AdversialClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                    ##                 str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                    ##if not instance.get( biBurstDataKey ):
                    ##    instance[biBurstDataKey] = 0
                    ##instance[biBurstDataKey] += 1

                    # PairBurst (up_down) (0_1)
                    if prevDirectionCursor==1 and directionCursor==0:
                        pairBurstDataKey = 'Pair-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                        if not instance.get( pairBurstDataKey ):
                            instance[pairBurstDataKey] = 0
                        instance[pairBurstDataKey] += 1

                    # time
                    '''
                    biBurstTimeKey = 'BiTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( biBurstTimeKey ):
                        instance[biBurstTimeKey] = 0
                    instance[biBurstTimeKey] += 1
                    '''

                    # PairBurst (up_down) (0_1)
                    if prevDirectionCursor==1 and directionCursor==0:
                        pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                         str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                        if not instance.get( pairBurstTimeKey ):
                            instance[pairBurstTimeKey] = 0
                        instance[pairBurstTimeKey] += 1

                prevDirectionCursor = directionCursor

                directionCursor = packet.getDirection()

                prevDataCursor = dataCursor
                dataCursor     = 0

                prevTimeCursor = timeCursor

                timeCursor = 0

                secondBurstAndUp = True

            dataCursor += packet.getLength()
            timeCursor = packet.getTime() - burstTimeRef

        if dataCursor>0:
            key = 'S'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
            if not instance.get( key ):
                instance[key] = 0
            instance[key] += 1

            timeKey = 'T'+str(directionCursor)+'-'+str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
            if not instance.get( timeKey ):
                instance[timeKey] = 0
            instance[timeKey] += 1

            burstTimeList.append(timeCursor)

            if (directionCursor==0):
                upBurstTimeList.append(timeCursor)

            if (directionCursor==1):
                #if timeCursor <> 0: # to be removed Jun 1, 15
                downBurstTimeList.append(timeCursor)

            # BiBurst
            if secondBurstAndUp:
                ##biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                ##                 str( AdversialClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                ##                 str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                ##if not instance.get( biBurstDataKey ):
                ##    instance[biBurstDataKey] = 0
                ##instance[biBurstDataKey] += 1

                # PairBurst (up_down) (1_0)
                if prevDirectionCursor==1 and directionCursor==0:
                    pairBurstDataKey = 'Pair-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(dataCursor, 600) )
                    if not instance.get( pairBurstDataKey ):
                        instance[pairBurstDataKey] = 0
                    instance[pairBurstDataKey] += 1

                # time
                '''
                biBurstTimeKey = 'BiTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                 str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                 str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                if not instance.get( biBurstTimeKey ):
                    instance[biBurstTimeKey] = 0
                instance[biBurstTimeKey] += 1
                '''

                # PairBurst (up_down) (1_0)
                if prevDirectionCursor==1 and directionCursor==0:
                    pairBurstTimeKey = 'PairTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(prevTimeCursor, timeBase) )+'-'+ \
                                     str( AdversialClassifier.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( pairBurstTimeKey ):
                        instance[pairBurstTimeKey] = 0
                    instance[pairBurstTimeKey] += 1

        instance['bandwidthUp'] = trace.getBandwidth( Packet.UP )
        instance['bandwidthDown'] = trace.getBandwidth( Packet.DOWN )

        maxTime = 0
        for packet in trace.getPackets():
             if packet.getTime() > maxTime:
                 maxTime = packet.getTime()
        instance['time'] = maxTime

        # Five number summary
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
            #print c.min(), ' ', np.percentile(c, 25), ' ', np.percentile(c, 50), ' ', np.percentile(c, 75), ' ', c.max()
            #print c
        else:
            instance['downBurstTimeMin']    = 0
            instance['downBurstTimeLowerQuartile']    = 0
            instance['downBurstTimeMedian'] = 0
            instance['downBurstTimeUpperQuartile']    = 0
            instance['downBurstTimeMax']    = 0

        instance['class'] = 'webpage'+str(trace.getId())
        return instance

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
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