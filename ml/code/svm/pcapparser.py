# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import Packet
import trace
import os.path
import glob
import config
import os
from Packet import Packet
from Trace import Trace
import dpkt
import random
import sys

def readfile( month, day, hour, webpageId ):
    strId = '.'.join([str(month), str(day), str(hour), str(webpageId)])

    trace = Trace(webpageId)
    start = 0

    absPath    = __constructAbsolutePath( month, day, hour, webpageId )

    # testing
    #print absPath

    if absPath:
        pcapReader = dpkt.pcap.Reader( file( absPath, "rb") )

        for ts, buf in pcapReader:
            eth = dpkt.ethernet.Ethernet(buf)
            ip  = eth.data
            tcp = ip.data
            
            if start==0: start = ts
            direction = Packet.UP
            if (tcp.sport==22):
                direction = Packet.DOWN
            if (config.DATA_SOURCE==3):
                if (tcp.sport==9001 or tcp.sport==443):
                    direction = Packet.DOWN
            if (config.DATA_SOURCE==4 or config.DATA_SOURCE==41 or config.DATA_SOURCE==42):
                if (tcp.sport==8080 or tcp.sport==443):
                    direction = Packet.DOWN
            '''
            #testing
            origTimeDiff = ts - start
            origTimeDiff = (ts - start) * 1000
            origTimeDiff = round(((ts - start) * 1000),0)
            origTimeDiff = int(round(((ts - start) * 1000),0))
            print origTimeDiff
            '''


            delta     = int(round(((ts - start) * 1000),0))
            length    = ip.len + Packet.HEADER_ETHERNET

            '''
            if (config.DATA_SOURCE==3): # overcoming the packet size greater than 1500
                while True:
                    if length > 1500: # MTU
                        excludedLength = random.randint(595, 1500)
                        trace.addPacket( Packet(direction, delta, excludedLength ) )
                        length = length - excludedLength
                    else:
                        trace.addPacket( Packet(direction, delta, length ) )
                        break
            else:
                trace.addPacket( Packet(direction, delta, length ) )
            '''
            trace.addPacket( Packet(direction, delta, length ) )

    return trace

def __constructAbsolutePath( month, day, hour, webpageId ):
    if not os.path.exists(config.PCAP_ROOT):
        raise Exception('Directory ('+config.PCAP_ROOT+') does not exist')
    
    monthStr = '%02d' % month
    dayStr   = '%02d' % day
    hourStr  = '%02d' % hour
    path     = os.path.join(config.PCAP_ROOT, '2006-'+monthStr
                                                 +'-'+dayStr
                                                 +'T'+hourStr
                                                 +'*/*'
                                                 +'-'+str(webpageId))
    if config.DATA_SOURCE == 3:
        path     = os.path.join(config.PCAP_ROOT, '2015-'+monthStr
                                                 +'-'+dayStr
                                                 +'T'+hourStr
                                                 +'*/*'
                                                 +'-'+str(webpageId)
                                                 +'*')

    if config.DATA_SOURCE == 4:
        path     = os.path.join(config.PCAP_ROOT, '2016-'+monthStr
                                                 +'-'+dayStr
                                                 +'T'+hourStr
                                                 +'*/*'
                                                 +'-'+str(webpageId)
                                                 +'*')

    if config.DATA_SOURCE == 41:
        if webpageId < 10000: # monitored from finance
            path     = os.path.join(config.PCAP_ROOT, '2016-'+monthStr
                                                 +'-'+dayStr
                                                 +'T'+hourStr
                                                 +'*/*'
                                                 +'-'+str(webpageId)
                                                 +'*')
        else: # unmonitored from communication
            AppsComPath = os.path.join(config.BASE_DIR   ,'pcap-logs-android-apps-communication')
            path     = os.path.join(AppsComPath, '2016-'
                                                 +'*'
                                                 +'-'+str(webpageId)
                                                 +'*')

    if config.DATA_SOURCE == 42:
        if webpageId < 10000: # monitored from finance
            path     = os.path.join(config.PCAP_ROOT, '2016-'+monthStr
                                                 +'-'+dayStr
                                                 +'T'+hourStr
                                                 +'*/*'
                                                 +'-'+str(webpageId)
                                                 +'*')
        else: # unmonitored from social
            AppsComPath = os.path.join(config.BASE_DIR   ,'pcap-logs-android-apps-social')
            path     = os.path.join(AppsComPath, '2016-'
                                                 +'*'
                                                 +'-'+str(webpageId)
                                                 +'*')
    pathList    =  glob.glob(path)

    absFilePath = None
    if len(pathList)>0:
        absFilePath = pathList[0]

    return absFilePath


def readfileHoneyPatch( webpageId, traceIndex ):
    trace = Trace(webpageId)
    start = 0

    if config.DATA_SOURCE == 6:
        absPath    = __constructAbsolutePathHoneyPatch( webpageId, traceIndex )
    elif config.DATA_SOURCE == 61:
        absPath    = __constructAbsolutePathHoneyPatchMultiClass( webpageId, traceIndex )
    elif config.DATA_SOURCE == 62 or config.DATA_SOURCE == 63 or config.DATA_SOURCE == 64:
        absPath    = __constructAbsolutePathHoneyPatchMultiClassAttackBenign( webpageId, traceIndex )

    if absPath:
        try:
            pcapReader = dpkt.pcap.Reader( file( absPath, "rb") )
        except:
            print absPath + ' has a problem reading file, training'
            return trace
            #sys.exit(2)

        #print absPath + ' is ok'

        firstTCP = False
        firstPacketkPort = 70000
        try:
            for ts, buf in pcapReader:
                eth = dpkt.ethernet.Ethernet(buf)
                ip  = eth.data
                tcp = ip.data

                if "TCP" not in str(type(ip.data)) or tcp.sport == 80 or tcp.dport == 80:
                    continue

                if firstTCP == False:
                    firstTCP = True
                    firstPacketkPort = tcp.sport # uplink

                if start==0: start = ts

                direction = Packet.UP

                if (tcp.sport==22):
                    direction = Packet.DOWN
                if (config.DATA_SOURCE==3):
                    if (tcp.sport==9001 or tcp.sport==443):
                        direction = Packet.DOWN
                if (config.DATA_SOURCE==4 or config.DATA_SOURCE==41 or config.DATA_SOURCE==42 or config.DATA_SOURCE==6 or config.DATA_SOURCE==61 or config.DATA_SOURCE==62 or config.DATA_SOURCE==63 or config.DATA_SOURCE==64):
                    #if (tcp.sport==8080 or tcp.sport==443):
                    if tcp.sport == firstPacketkPort:
                        direction = Packet.UP
                    else:
                        direction = Packet.DOWN

                delta     = int(round(((ts - start) * 1000),0))
                length    = ip.len + Packet.HEADER_ETHERNET

                trace.addPacket( Packet(direction, delta, length ) )
        except:
            print 'file ' + absPath + ' has a problem.'

    return trace

def readfileHoneyPatchSomePackets( webpageId, traceIndex ):
    trace = Trace(webpageId)
    start = 0

    if config.DATA_SOURCE == 6:
        absPath    = __constructAbsolutePathHoneyPatch( webpageId, traceIndex )
    elif config.DATA_SOURCE == 61:
        absPath    = __constructAbsolutePathHoneyPatchMultiClass( webpageId, traceIndex )
    elif config.DATA_SOURCE == 62 or config.DATA_SOURCE == 63 or config.DATA_SOURCE == 64:
        absPath    = __constructAbsolutePathHoneyPatchMultiClassAttackBenign( webpageId, traceIndex )

    packetCtr = 0

    if absPath:
        try:
            pcapReader = dpkt.pcap.Reader( file( absPath, "rb") )
        except:
            print absPath + ' has a problem reading file, testing'
            return trace
            #sys.exit(2)

        #print absPath + ' is ok'

        firstTCP = False
        firstPacketkPort = 70000

        try:
            for ts, buf in pcapReader:
                eth = dpkt.ethernet.Ethernet(buf)
                ip  = eth.data
                tcp = ip.data

                if "TCP" not in str(type(ip.data)) or tcp.sport == 80 or tcp.dport == 80:
                    continue

                if firstTCP == False:
                    firstTCP = True
                    firstPacketkPort = tcp.sport # uplink

                if start==0: start = ts
                direction = Packet.UP

                if (tcp.sport==22):
                    direction = Packet.DOWN
                if (config.DATA_SOURCE==3):
                    if (tcp.sport==9001 or tcp.sport==443):
                        direction = Packet.DOWN
                if (config.DATA_SOURCE==4 or config.DATA_SOURCE==41 or config.DATA_SOURCE==42 or config.DATA_SOURCE==6 or config.DATA_SOURCE==61 or config.DATA_SOURCE==62 or config.DATA_SOURCE==63 or config.DATA_SOURCE == 64):
                    #if (tcp.sport==8080 or tcp.sport==443):
                        #direction = Packet.DOWN
                    if tcp.sport == firstPacketkPort:
                        direction = Packet.UP
                    else:
                        direction = Packet.DOWN

                delta     = int(round(((ts - start) * 1000),0))
                length    = ip.len + Packet.HEADER_ETHERNET

                trace.addPacket( Packet(direction, delta, length ) )

                packetCtr += 1
                # Some packets from the trace are added
                if config.NUM_TRACE_PACKETS != -1 and packetCtr >= config.NUM_TRACE_PACKETS:
                    break
        except:
            print 'file ' + absPath + ' has a problem.'

    return trace

def __constructAbsolutePathHoneyPatch( webpageId, traceIndex ):
    if not os.path.exists(config.PCAP_ROOT):
        raise Exception('Directory ('+config.PCAP_ROOT+') does not exist')


    if webpageId == 0: # benign
        fileName = 'benign/' + 'stream-' + str(traceIndex) + '.cap'
    elif webpageId == 1: # attack
        fileName = 'attack/' + 'stream-' + str(traceIndex) + '.cap'

    path     = os.path.join(config.PCAP_ROOT, fileName)

    pathList    =  glob.glob(path)

    absFilePath = None
    if len(pathList)>0:
        absFilePath = pathList[0]

    return absFilePath

def __constructAbsolutePathHoneyPatchMultiClass( webpageId, traceIndex ):
    if not os.path.exists(config.PCAP_ROOT):
        raise Exception('Directory ('+config.PCAP_ROOT+') does not exist')


    if webpageId == 0: # benign
        #fileName = 'benign2/' + 'stream-' + str(traceIndex) + '.cap'
        fileName = 'benignWordpress/' + 'stream-' + str(traceIndex) + '.cap'
        #fileName = 'benign2/' + 'stream-' + str(traceIndex) + '.cap'

    else: # attack1, attack2, ...etc
        fileName = 'attack'+ str(webpageId) + '/' + 'stream-' + str(traceIndex) + '.cap'

    path     = os.path.join(config.PCAP_ROOT, fileName)

    pathList    =  glob.glob(path)

    absFilePath = None
    if len(pathList)>0:
        absFilePath = pathList[0]

    return absFilePath

def __constructAbsolutePathHoneyPatchMultiClassAttackBenign( webpageId, traceIndex ):
    if not os.path.exists(config.PCAP_ROOT):
        raise Exception('Directory ('+config.PCAP_ROOT+') does not exist')


    if webpageId < config.NUM_BENIGN_CLASSES: # -i: used in closed-world and open-world to decide number of benign classes
        fileName = 'benign' + str(webpageId) + '/' + 'stream-' + str(traceIndex) + '.cap'

    else: # attacksX, X >= config.NUM_BENIGN_CLASSES
        fileName = 'attack' + str(webpageId) + '/' + 'stream-' + str(traceIndex) + '.cap'

    path     = os.path.join(config.PCAP_ROOT, fileName)

    pathList    =  glob.glob(path)

    absFilePath = None
    if len(pathList)>0:
        absFilePath = pathList[0]

    return absFilePath