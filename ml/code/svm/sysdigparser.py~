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
import subprocess
from Event import Event
import datetime
import math
from EventTrace import EventTrace
import signal

def readfileHoneyPatch( webpageId, traceIndex ):
    #trace = Trace(webpageId)
    eventTrace = EventTrace(webpageId, traceIndex)

    if config.DATA_SOURCE == 65:
        absPath    = __constructAbsolutePathHoneyPatchSysdig( webpageId, traceIndex )

    if absPath:
        ###print absPath + ':  ' + str(webpageId) + '_' + str(traceIndex)
        #cmd = 'sysdig -r ' + absPath
        '''
        file = os.path.join(config.SYSDIG, 'tempSysdig')
        sysdigargs = "proc.name=apache2"
        numLines = 3000
        if config.CLASSIFIER == 33:
            # kNN-LCS
            cmd = 'sysdig -r ' + absPath + ' "(evt.type=read or evt.type=write or evt.type=creat or evt.type=open or \
                                               evt.type=close or evt.type=stat or evt.type=fstat) and evt.dir=\'>\'" ' + ' | head -n ' + str(numLines) + ' > ' + file
            #cmd = 'sysdig -r ' + absPath + ' "(evt.type=read or evt.type=write) and evt.dir=\'>\'" ' + ' > ' + file
        else:
            cmd = 'sysdig -r ' + absPath + ' | head -n ' + str(numLines) + ' > ' + file
            #cmd = 'sysdig -r ' + absPath + ' " evt.dir=\'>\'" ' + ' | head -n ' + str(numLines) + ' > ' + file
            #cmd = 'sysdig -r ' + absPath + ' > ' + file
        #pp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #pp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid)
        pp = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        #pp = subprocess.Popen(cmd, shell=True)
        pp.wait()

        fileLines = [line.strip() for line in open(file)]
        '''

        fileLines = [line.strip() for line in open(absPath)]
        #for line in pp.stdout:
        for line in fileLines:
            # every line is an event (syscall)
            line = line.rstrip()
            lineArray = line.split(" ")

            if len(lineArray) < 7:
                continue


            #print str(lineArray[3])
            if str(lineArray[3]).strip() == 'tcpdump': # eliminating irrelevant process name
                continue

            '''
            number = int(lineArray[0])

            timeStr = str(lineArray[1]) # 16:58:18.038556857
            inHour = timeStr[:2]
            inMin = timeStr[3:5]
            inSec = timeStr[6:8]
            inNanosec = int(timeStr[-9:])
            inMicrosec = math.floor(inNanosec/1000.0)
            time = datetime.time(int(inHour), int(inMin), int(inSec), int(inMicrosec)).strftime('%s')
            cpu = int(lineArray[2])
            processname = str(lineArray[3])
            threadid = int(lineArray[4].split("(")[1].split(")")[0]) # (xxxxx)
            '''
            direction = Event.ENTER_EVT
            if str(lineArray[5]) == '<':
                direction = Event.EXIT_EVT

            systemcallName = str(lineArray[6])


            # Some sysdig events are not system calls, so don't consider such events

            if systemcallName not in config.sysCallTable:
                continue

            '''
            if len(lineArray) > 7:
                args = ' '.join(lineArray[7:])
            else:
                args = ""
            '''
            args = "" # temporarily
            #event = Event(number, time, cpu, processname, threadid, direction, systemcallName, args)

            #trace.addPacket( Packet(event.getDirection(), event.getTime(), event.getSystemcallIndex()) )

            #eventTrace.addEvent( Event(number, time, cpu, processname, threadid, direction, systemcallName, args) )
            eventTrace.addEvent( Event(direction, systemcallName) )

        '''
        sysdigStatFilename = os.path.join(config.SYSDIG,'sysdigStats-'+str(eventTrace.getId())+'-'+str(eventTrace.getTraceIndex()))
        f = open( sysdigStatFilename+'.sysdig', 'a' )
        print 'EventTrace number: ' + str(eventTrace.getId())
        print 'File name: ' + absPath
        f.write( 'EventTrace number: ' + str(eventTrace.getId()) )
        f.write('\n\n')
        f.write( 'File name: ' + absPath )
        f.write('\n\n')
        f.write(  str(eventTrace.getNGramHistogram(N=1,sortReverseByValue=True,plotList=True)) )
        f.write('\n\n')
        f.write(  str(eventTrace.getNGramHistogram(N=2,sortReverseByValue=True,plotList=True)) )
        f.write('\n\n')
        f.close()
        '''

        #print 'kill process...'

        #print 'size of pp.stdout: ' + str(pp.stdout.__sizeof__())

        #os.killpg(os.getpgid(pp.pid), signal.SIGTERM)
        #pp.kill()
        #pp.terminate()

        #return trace
        return eventTrace


def __constructAbsolutePathHoneyPatchSysdig( webpageId, traceIndex ):
    if not os.path.exists(config.PCAP_ROOT):
        raise Exception('Directory ('+config.PCAP_ROOT+') does not exist')


    if webpageId < config.NUM_BENIGN_CLASSES: # -i: used in closed-world and open-world to decide number of benign classes
        fileName = 'benign' + str(webpageId) + '/' + 'stream-' + str(traceIndex) + '.scap'

    else: # attacksX, X >= config.NUM_BENIGN_CLASSES
        fileName = 'attack' + str(webpageId) + '/' + 'stream-' + str(traceIndex) + '.scap'

    path     = os.path.join(config.PCAP_ROOT, fileName)

    pathList    =  glob.glob(path)

    absFilePath = None
    if len(pathList)>0:
        absFilePath = pathList[0]

    return absFilePath