
import math
from Event import Event
import operator
import config
import os
import time

class EventTrace:
    def __init__(self,id, traceIndex):
        self.__eventArray   = []
        self.__id            = id
        self.__histogramEnter   = {}
        self.__histogramExit = {}
        self.__enterEvent     = 0
        self.__exitEvent   = 0
        self.__traceIndex = traceIndex

    #def __del__(self):
    #    print 'event trace ' + str(self.__id) + ' died'

    def getId(self): return self.__id

    def setId(self,id):
        self.__id = id

    def getTraceIndex(self): return self.__traceIndex

    def setTraceIndex(self,traceIndex):
        self.__traceIndex = traceIndex

    def getEventCount( self, direction = None ):
        return len(self.getEvents(direction))

    def getEvents( self, direction = None ):
        retArray = []
        for event in self.__eventArray:
            if direction == None or event.getDirection() == direction:
                retArray.append( event )
        return retArray

    def addEvent( self, event ):

        key = str(event.getDirection())+'-'+str(event.getSystemcallIndex())

        if event.getDirection()==event.ENTER_EVT:
            self.__enterEvent += 1
            if not self.__histogramEnter.get( key ):
                self.__histogramEnter[key] = 0
            self.__histogramEnter[key] += 1
        elif event.getDirection()==event.EXIT_EVT:
            self.__exitEvent += 1
            if not self.__histogramExit.get( key ):
                self.__histogramExit[key] = 0
            self.__histogramExit[key] += 1

        return self.__eventArray.append( event )

    def getTime( self, direction = None ):
        timeCursor = 0
        for event in self.getEvents():
            if direction == None or direction == event.getDirection():
                timeCursor = event.getTime()

        return timeCursor

    # Same name (getBandwidth) as method used in Trace (num bytes) and EventTrace (num events)
    def getBandwidth( self, direction = None ):
        count = 0
        try:
            count = self.getEventCount()
        except:
            print 'Error getting bandwidth in trace: ' + str(self.getTraceIndex()) + ' for benign/attack ' + str(self.getId())

        return count

    def getHistogram( self, direction = None, normalize = False ):
        if direction == Event.ENTER_EVT:
            histogram = dict(self.__histogramEnter)
            totalEvents = self.__enterEvent
        elif direction == Event.EXIT_EVT:
            histogram = dict(self.__histogramExit)
            totalEvents = self.__exitEvent
        else:
            histogram = dict(self.__histogramEnter)
            for key in self.__histogramExit:
                histogram[key] = self.__histogramExit[key]
            totalEvents = self.__exitEvent + self.__enterEvent

        if normalize==True:
            for key in histogram:
                histogram[key] = (histogram[key] * 1.0) / totalEvents

        return histogram

    def calcL1Distance( self, targetDistribution, filterDirection=None ):
        localDistribution  = self.getHistogram( filterDirection, True )

        keys = localDistribution.keys()
        for key in targetDistribution:
            if key not in keys:
                keys.append( key )

        distance = 0
        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None and r == None: continue
            if l == None: l = 0
            if r == None: r = 0

            distance += math.fabs( l - r )

        return distance

    def getMostSkewedDimension( self, targetDistribution ):
        localDistribution  = self.getHistogram( None, True )

        keys = targetDistribution.keys()

        worstKey = None
        worstKeyDistance = 0

        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None: l = 0
            if r == None: r = 0

            if worstKey==None or (r - l) > worstKeyDistance:
                worstKeyDistance = r - l
                worstKey = key

        bits = worstKey.split('-')

        return [int(bits[0]),int(bits[1])]

    def get2GramHistogram( self, direction = None, normalize = False ):
        currEvent = nextEvent = self.__eventArray[0]
        histogram = {}
        for i in range(1,len(self.__eventArray)):
            nextEvent = self.__eventArray[i]
            key = currEvent.getSystemcallName + '-' + nextEvent.getSystemcallName
            if not histogram.get( key ):
                histogram[key] = 0
            histogram[key] += 1
            currEvent = nextEvent

        return histogram


    def getNGramHistogram( self, direction = None, normalize = False, N = 2, sortReverseByValue=False, plotList=False ):
        histogram = {}
        if len(self.__eventArray) == 0:
            # sysdig trace may be empty
            return histogram

        NGramEvents = [] # holds curr, next, nextnext, ...
        for i in range(0,N):
            NGramEvents.append(self.__eventArray[0]) # Initially, assign all to the first event in the sysdig file

        breakOuter = False

        for e in range(0,len(self.__eventArray)):

            ei = 0 # for next event, next next event, ...
            for i in range(1,N):
                ei += 1
                if (e + ei) == len(self.__eventArray):
                    # break the inner loop
                    breakOuter = True
                    break

                NGramEvents[i] = self.__eventArray[e+ei] # nextEvent

            if breakOuter:
                # break the outer loop
                break

            key = ''
            for i in range(0,N):
                #key = key + NGramEvents[i].getSystemcallName() + '-'
                key = key + str(NGramEvents[i].getSystemcallIndex()) + '-'
            key = key[:-1] # get rid of '-'

            if not histogram.get( key ):
                histogram[key] = 0
            histogram[key] += 1

            for i in range(0,N):
                if (e + 1) < len(self.__eventArray):
                    NGramEvents[i] = self.__eventArray[e+1] # assign all to event e+1 (next)


        if sortReverseByValue==False:
            tuplesList = list(histogram.items())
        elif sortReverseByValue==True:
            tuplesList =  sorted(histogram.items(), key=operator.itemgetter(1), reverse=True) # returns tuples

        if plotList:
            self.draw(tuplesList[:20], N)

        return tuplesList

    def draw(self, tuplesList, N):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 5})
        # save the names and their respective scores separately
        # reverse the tuples to go from most frequent to least frequent
        x = zip(*tuplesList)[0]
        y = zip(*tuplesList)[1]
        x_pos = np.arange(len(x))

        # calculate slope and intercept for the linear trend line
        slope, intercept = np.polyfit(x_pos, y, 1)
        trendline = intercept + (slope * x_pos)

        plt.plot(x_pos, trendline, color='red', linestyle='--')
        plt.bar(x_pos, y,align='center')
        plt.xticks(x_pos, x, rotation='30')
        plt.ylabel('Frequency')

        if self.getId() < config.NUM_BENIGN_CLASSES:
            print 'benign' + str(self.getId()) + '-' + str(self.getTraceIndex()) + '-' + str(N) + 'gram.pdf'
            plt.savefig(os.path.join(config.SYSDIG, 'benign' + str(self.getId()) + '-' + str(self.getTraceIndex()) + '-' + str(N) + 'gram.png'))
            #time.sleep(10)
        else:
            print 'attack' + str(self.getId()) + '-' + str(self.getTraceIndex()) + '-' + str(N) + 'gram.pdf'
            plt.savefig(os.path.join(config.SYSDIG, 'attack' + str(self.getId()) + '-' + str(self.getTraceIndex()) + '-' + str(N) + 'gram.png'))
            #time.sleep(10)

        plt.show()
        #time.sleep(2)
        #plt.close('all')


    def getNGramHistogramOnePass( self, direction = None, normalize = False, N = 2, sortReverseByValue=False, plotList=False ):
        '''
        histogramsList = []
        for i in range(0,N):
            #histogram = {}
            histogramsList.append({}) # histogramsList[0] is 1 gram histogram

        if len(self.__eventArray) == 0:
            # sysdig trace may be empty
            #return histogram
            return histogramsList

        NGramEvents = [] # holds curr, next, nextnext, ...
        for i in range(0,N):
            NGramEvents.append(self.__eventArray[0]) # Initially, assign all to the first event in the sysdig file

        breakOuter = False

        for e in range(0,len(self.__eventArray)):

            ei = 0 # for next event, next next event, ...
            for i in range(1,N):
                ei += 1
                if (e + ei) == len(self.__eventArray):
                    # break the inner loop
                    breakOuter = True
                    break

                NGramEvents[i] = self.__eventArray[e+ei] # nextEvent

            if breakOuter:
                # break the outer loop
                break

            key = ''
            for i in range(0,N):
                #key = key + NGramEvents[i].getSystemcallName() + '-'
                key = key + str(NGramEvents[i].getSystemcallIndex()) + '-'
            key = key[:-1] # get rid of '-'

            if not histogram.get( key ):
                histogram[key] = 0
            histogram[key] += 1

            for i in range(0,N):
                if (e + 1) < len(self.__eventArray):
                    NGramEvents[i] = self.__eventArray[e+1] # assign all to event e+1 (next)


        if sortReverseByValue==False:
            tuplesList = list(histogram.items())
        elif sortReverseByValue==True:
            tuplesList =  sorted(histogram.items(), key=operator.itemgetter(1), reverse=True) # returns tuples

        if plotList:
            self.draw(tuplesList[:20], N)

        return tuplesList
        '''
        pass