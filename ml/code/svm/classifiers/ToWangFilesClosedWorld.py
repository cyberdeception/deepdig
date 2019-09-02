import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math

import numpy as np

import config
import os

class ToWangFilesClosedWorld:
    @staticmethod
    def traceToInstances( trace, webpageIndex, entity, OSADfolder ): # trace to packets to write OSAD files
        fileSb = []
        for packet in trace.getPackets():
            direction = ''
            if packet.getDirection() == 1:
                direction = '-'
            #fileSb.append(direction+str(packet.getLength()))
            fileSb.append(direction+str(101)) # since OSAD experct sizes as 101 not 1

        filename = str(webpageIndex)+"_"+str(entity)+".txt"
        f = open(os.path.join(OSADfolder, filename), 'w')
        f.write("\n".join(fileSb ))
        f.close()