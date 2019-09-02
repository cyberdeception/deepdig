# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

#import MySQLdb
import math
import config
import pcapparser

from Webpage import Webpage
from Trace import Trace
from Packet import Packet

import memcache
mc = memcache.Client(['127.0.0.1:11211'], debug=0)
ENABLE_CACHE = False

import cPickle
import os
from Utils import Utils

import random
import sysdigparser
import sys

class Datastore:
    @staticmethod
    def getWebpagesLL( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                trace = Datastore.getTraceLL( webpageId, traceIndex )
                webpage.addTrace(trace)
            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceLL( webpageId, traceIndex ):
        key = '.'.join(['Webpage',
                        'LL',
                        str(webpageId),
                        str(traceIndex)])

        trace = mc.get(key)
        if ENABLE_CACHE and trace:
            trace = cPickle.loads(trace)
        else:
            dateTime = config.DATA_SET[traceIndex]
            trace = pcapparser.readfile(dateTime['month'],
                                        dateTime['day'],
                                        dateTime['hour'],
                                        webpageId)

            mc.set(key,cPickle.dumps(trace,protocol=cPickle.HIGHEST_PROTOCOL))

        return trace

    @staticmethod
    def getWebpagesHerrmann( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                trace = Datastore.getTraceHerrmann( webpageId, traceIndex )
                webpage.addTrace(trace)
            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceHerrmann( webpageId, traceIndex ):
        if config.DATA_SOURCE == 1:
            datasourceId = 4
        elif config.DATA_SOURCE == 2:
            datasourceId = 5

        key = '.'.join(['Webpage',
                        'H',
                        str(datasourceId),
                        str(webpageId),
                        str(traceIndex)])

        trace = mc.get(key)
        if ENABLE_CACHE and trace:
            trace = cPickle.loads(trace)
        else:
            connection = MySQLdb.connect(host=config.MYSQL_HOST,
                                         user=config.MYSQL_USER,
                                         passwd=config.MYSQL_PASSWD,
                                         db=config.MYSQL_DB )

            cursor = connection.cursor()
            command = """SELECT packets.trace_id,
                                      packets.size,
                                      ROUND(packets.abstime*1000)
                                 FROM (SELECT id
                                         FROM traces
                                        WHERE site_id = (SELECT id
                                                           FROM sites
                                                          WHERE dataset_id = """+str(datasourceId)+"""
                                                          ORDER BY id
                                                          LIMIT """+str(webpageId)+""",1)
                                        ORDER BY id
                                        LIMIT """+str(traceIndex)+""",1) traces,
                                      packets
                                WHERE traces.id = packets.trace_id
                                ORDER BY packets.trace_id, packets.abstime"""
            cursor.execute( command )

            data = cursor.fetchall()
            trace = Trace(webpageId)
            for item in data:
                direction = Packet.UP
                if int(item[1])>0:
                    direction = Packet.DOWN
                time   = item[2]
                length = int(math.fabs(item[1]))

                trace.addPacket( Packet( direction, time, length ) )
            connection.close()

            mc.set(key,cPickle.dumps(trace,protocol=cPickle.HIGHEST_PROTOCOL))

        return trace


    @staticmethod
    def getWebpagesWangTor( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                trace = Datastore.getTraceWangTor( webpageId, traceIndex ) # For monitored websites: read file batch/webpageId-traceIndex
                webpage.addTrace(trace)

            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceWangTor( webpageId, traceIndex ):

        trace = Datastore.readWangTorFile(webpageId, traceIndex)

        return trace

    @staticmethod
    def readWangTorFile( webpageId, traceIndex ):

        if webpageId < 100: # 100 and more is nonMon
            file = os.path.join(config.PCAP_ROOT, str(webpageId)+"-"+str(traceIndex))

        else:
            file = os.path.join(config.PCAP_ROOT, str(webpageId-100)) # as the nonMon id starts from 100 and the file names are 0, 1, 2, 3, ...

        fileList = Utils.readFile(file)

        trace = Trace(webpageId)

        '''
        0.0	1 cell
        0.0	1
        0.116133928299	1
        0.499715805054	-1
        0.499715805054	-1
        ...
        '''

        for i in range(1,len(fileList)):
            cellArray = fileList[i].split("\t")
            cTime = cellArray[0]
            cDirection = int(cellArray[1])

            pDirection = Packet.UP
            if (cDirection==-1):
                pDirection = Packet.DOWN

            # as in the pcapparser.py
            # delta     = int(round(((ts - start) * 1000),0))
            pTime = int(round((float(cTime) * 1000),0))

            pLength = abs(int(cellArray[1])) # sizes are only 1 and -1

            trace.addPacket(Packet(pDirection, pTime, pLength))


        return trace

    @staticmethod
    def readWangTorFileOld( webpageId, traceIndex ):

        if webpageId < 100:
            file = os.path.join(config.PCAP_ROOT, str(webpageId)+"-"+str(traceIndex))

        else:
            file = os.path.join(config.PCAP_ROOT, str(webpageId-100)) # as the nonMon id starts from 100 and the file names are 0, 1, 2, 3, ...

        fileList = Utils.readFile(file)

        trace = Trace(webpageId)

        '''
        0.0	1 cell
        0.0	1
        0.116133928299	1
        0.499715805054	-1
        0.499715805054	-1
        ...
        '''

        prevcTime = currcTime = fileList[0].split("\t")[0] # previous and current cell time (0.0 in the example above)
        prevcDirection = currcDirection = int(fileList[0].split("\t")[1]) # previous and current cell direction

        #cLength = 512 # cell length is always 512 bytes in Tor
        cellCtr = 1

        for i in range(1,len(fileList)):
            cellArray = fileList[i].split("\t")
            cTime = cellArray[0]
            cDirection = cellArray[1]

            currcTime = cTime
            currcDirection = int(cDirection)

            if currcTime != prevcTime:
                #pLength = cellCtr * cLength
                Datastore.addPacketsFromCells( trace, prevcDirection, prevcTime, cellCtr )
                prevcDirection = currcDirection
                prevcTime = currcTime
                cellCtr = 1
                continue
            elif currcDirection != prevcDirection:
                #pLength = cellCtr * cLength
                Datastore.addPacketsFromCells( trace, prevcDirection, prevcTime, cellCtr )
                prevcDirection = currcDirection
                prevcTime = currcTime
                cellCtr = 1
                continue
            else: # same time, same direciton
                cellCtr = cellCtr + 1
                prevcDirection = currcDirection
                prevcTime = currcTime

        # for the last cell
        Datastore.addPacketsFromCells( trace, prevcDirection, prevcTime, cellCtr )

        return trace

    @staticmethod
    def addPacketsFromCellsOld2(trace, prevcDirection, prevcTime, cellCtr):
        '''
        cellCtr: number of cells in the same direction (1's of -1's) and same time
        when cellCtr = 1 then add 512
        when cellCtr = 2 then add 1024
        when cellCtr = 3 then add 1536 (then 1500 MTU)
        when cellCtr = 4 then add 1536 (1500) then 512
        when cellCtr = 5 then add 1536 (1500) then 1024
        when cellCtr = 6 then add 1536 (1500) twice
        and so on
        '''
        cLength = 512 # cell length is always 512 bytes in Tor

        pDirection = Packet.UP
        if (prevcDirection==-1):
            pDirection = Packet.DOWN

        # as in the pcapparser.py
        # delta     = int(round(((ts - start) * 1000),0))
        prevcTime = int(round((float(prevcTime) * 1000),0))

        while cellCtr >= 1:
            if cellCtr == 1:
                pLength = random.randint( 512, 1000 )
                trace.addPacket(Packet(pDirection, prevcTime, pLength)) # add 512
                #trace.addPacket(Packet(pDirection, prevcTime, cLength)) # add 512
                break
            elif cellCtr == 2:
                pLength = random.randint( 1024, 1200 )
                trace.addPacket(Packet(pDirection, prevcTime, pLength)) # add 1024
                #trace.addPacket(Packet(pDirection, prevcTime, 2*cLength)) # add 1024
                #cellCtr = cellCtr - 2
                break
            else:
                trace.addPacket(Packet(pDirection, prevcTime, 1500)) # 3*cLength = 1536 (or 1500 which is the MTU)
                cellCtr = cellCtr - 3

    @staticmethod
    def addPacketsFromCells(trace, prevcDirection, prevcTime, cellCtr):
        '''
        cellCtr: number of cells in the same direction (1's of -1's) and same time
        when cellCtr = 1 then add 512
        when cellCtr = 2 then add 1024
        when cellCtr = 3 then add 1536 (then 1500 MTU)
        when cellCtr = 4 then add 1536 (1500) then 512
        when cellCtr = 5 then add 1536 (1500) then 1024
        when cellCtr = 6 then add 1536 (1500) twice
        and so on
        '''
        cLength = 512 # cell length is always 512 bytes in Tor

        pDirection = Packet.UP
        if (prevcDirection==-1):
            pDirection = Packet.DOWN

        # as in the pcapparser.py
        # delta     = int(round(((ts - start) * 1000),0))
        prevcTime = int(round((float(prevcTime) * 1000),0))

        while cellCtr >= 1:
            if cellCtr == 1:
                trace.addPacket(Packet(pDirection, prevcTime, cLength)) # add 512
                break
            elif cellCtr == 2:
                trace.addPacket(Packet(pDirection, prevcTime, 2*cLength)) # add 1024
                #cellCtr = cellCtr - 2
                break
            else:
                trace.addPacket(Packet(pDirection, prevcTime, 1500)) # 3*cLength = 1536 (or 1500 which is the MTU)
                cellCtr = cellCtr - 3


    @staticmethod
    def addPacketsFromCellsOld(trace, prevcDirection, prevcTime, cellCtr):
        '''
        cellCtr: number of cells in the same direction (1's of -1's) and same time
        when cellCtr = 1 then add 512
        when cellCtr = 2 then add 1024
        when cellCtr = 3 then add 1024 and 512
        when cellCtr = 4 then add 1024 twice
        when cellCtr = 5 then add 1024 twice and one 512
        and so on
        '''
        cLength = 512 # cell length is always 512 bytes in Tor

        pDirection = Packet.UP
        if (prevcDirection==-1):
            pDirection = Packet.DOWN

        # as in the pcapparser.py
        # delta     = int(round(((ts - start) * 1000),0))
        prevcTime = int(round((float(prevcTime) * 1000),0))

        while cellCtr >= 1:
            if cellCtr == 1:
                trace.addPacket(Packet(pDirection, prevcTime, cLength)) # add 512
                break
            else:
                trace.addPacket(Packet(pDirection, prevcTime, 2*cLength)) # add 1024
                cellCtr = cellCtr - 2


    # Not used
    @staticmethod
    def getDummyWebpages(webpageId):
        dummyWebpages = []

        dummyWebpage = Webpage(webpageId)

        # add an empty trace
        #dummyTrace = Trace(webpageId)
        #dummyWebpage.addTrace(dummyTrace)
        dummyWebpage.addTrace([])

        dummyWebpages.append(dummyWebpage)

        return dummyWebpages


    @staticmethod
    def getWebpagesHoneyPatch( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                checkKey = str(webpageId) + '_' + str(traceIndex)
                if config.excludedInst.get( checkKey ):
                    print str(webpageId) + '_' + str(traceIndex) + ' removed'
                    continue
                trace = Datastore.getTraceHoneyPatch( webpageId, traceIndex ) # webpageId = {0, 1} and this specifies the folder {benign, attack}. traceIndex = x which is stream-x.pcap
                webpage.addTrace(trace)
            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceHoneyPatch( webpageId, traceIndex ):

        trace = pcapparser.readfileHoneyPatch( webpageId, traceIndex )

        return trace

    @staticmethod
    def getWebpagesHoneyPatchSomePackets( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                checkKey = str(webpageId) + '_' + str(traceIndex)
                if config.excludedInst.get( checkKey ):
                    print str(webpageId) + '_' + str(traceIndex) + ' removed'
                    continue
                trace = Datastore.getTraceHoneyPatchSomePackets( webpageId, traceIndex ) # webpageId = {0, 1} and this specifies the folder {benign, attack}. traceIndex = x which is stream-x.pcap
                webpage.addTrace(trace)
            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceHoneyPatchSomePackets( webpageId, traceIndex ):

        trace = pcapparser.readfileHoneyPatchSomePackets( webpageId, traceIndex )

        return trace


###
    @staticmethod
    def getWebpagesHoneyPatchSysdigTest( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                trace = Datastore.getTraceHoneyPatchSysdig( webpageId, traceIndex ) # webpageId = {0, 1} and this specifies the folder {benign, attack}. traceIndex = x which is stream-x.pcap
                webpage.addTrace(trace)

            webpages.append(webpage)

        return webpages

    @staticmethod
    def getWebpagesHoneyPatchSysdig( webpageIds, traceIndexStart, traceIndexEnd ):
        webpages = []
        for webpageId in webpageIds:
            webpage = Webpage(webpageId)
            for traceIndex in range(traceIndexStart, traceIndexEnd):
                trace = Datastore.getTraceHoneyPatchSysdig( webpageId, traceIndex ) # webpageId = {0, 1} and this specifies the folder {benign, attack}. traceIndex = x which is stream-x.pcap

                if trace.getEventCount() != 0:
                    webpage.addTrace(trace)
                else:
                    key = str(webpageId) + '_' + str(traceIndex)
                    config.excludedInst[key] = 1
                    print str(webpageId) + '_' + str(traceIndex) + ' is empty.'

            webpages.append(webpage)

        return webpages

    @staticmethod
    def getTraceHoneyPatchSysdig( webpageId, traceIndex ):

        trace = sysdigparser.readfileHoneyPatch( webpageId, traceIndex )

        return trace
