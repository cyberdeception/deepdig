import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math

import numpy as np

import config
import os

class ToWangFilesOpenWorld:
    @staticmethod
    def traceToInstances( trace, entity, WangOpenWorldKnnfolder, monitoredWebpageIdsObj, unMonitoredWebpageIdsObj ): # trace to 1's and -1's
        #webpageIndex
        fileSb = [] # file string builder
        for packet in trace.getPackets():
            #print packet.getLength()
            direction = ''
            if packet.getDirection() == 1: # downlink
                direction = '-'
            '''
            num512Cells = packet.getLength() / 512;
            for i in range(num512Cells):
                fileSb.append(str(packet.getTime()) + '\t' + direction + '1')
            '''
            fileSb.append(str(packet.getTime()) + '\t' + direction + '1')

        #print 'cell list length ' + str(len(fileSb))

        if (len(fileSb) == 0 and monitoredWebpageIdsObj.__contains__(trace.getId())): # to overcome when a trace doesn't have enough packets to form cells
            print 'mon website ' + str(trace.getId()) + ', file '+ str(monitoredWebpageIdsObj.index(trace.getId())) + ' - cell list length ' + str(len(fileSb)) + ' ... bad sample'
            fileSb.append('1'+'\t'+'1')
            fileSb.append('2'+'\t'+'-1')
            #exit()

        if (len(fileSb) == 0  and unMonitoredWebpageIdsObj.__contains__(trace.getId())): # to overcome when a trace doesn't have enough packets to form cells
            print 'unmon website ' + str(trace.getId()) + ', file '+ str(unMonitoredWebpageIdsObj.index(trace.getId())) + ' - cell list length ' + str(len(fileSb)) + ' ... bad sample'
            fileSb.append('1'+'\t'+'1')
            fileSb.append('2'+'\t'+'-1')
            #print fileSb




        if monitoredWebpageIdsObj.__contains__(trace.getId()):
            # monitored
            # file name like 1-1
            filename = str(monitoredWebpageIdsObj.index(trace.getId()))+"-"+str(entity)
        elif unMonitoredWebpageIdsObj.__contains__(trace.getId()):
            # unmonitored
            # file name like 1,  Last instance of that class will override and we will have one file only (so one instance per unmonitored)
            filename = str(unMonitoredWebpageIdsObj.index(trace.getId()))
        else:
            print "Error, webpage is not in the monitored or unmonitored list. Open World. Wang files (1, -1)"

        f = open(os.path.join(WangOpenWorldKnnfolder+'/batch', filename), 'w')
        f.write("\n".join(fileSb ))
        f.close()

