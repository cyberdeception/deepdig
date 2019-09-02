

import unittest
import sysdigparser
import os
import config
from Datastore import Datastore
from EventTrace import EventTrace
from Event import Event

class SysdigParserTestCase(unittest.TestCase):
    def test_readfile(self):
        config.PCAP_ROOT = os.path.join(config.BASE_DIR   ,'honeypatckBenattackTest/sysdig')
        config.DATA_SOURCE = 65
        config.NUM_BENIGN_CLASSES = 12
        config.SYSDIG = './sysdigtest'

        #eventTrace = sysdigparser.readfileHoneyPatch( 5, 156 )

        #print str(eventTrace.getId())

        for traceId in range(12,30):
            #traceId = 5
            traceStart = 0
            traceEnd = 3

            webpage = Datastore.getWebpagesHoneyPatchSysdig([traceId], traceStart, traceEnd) # bug, files missing 157, 158, ...
            webpageTest = webpage[0]
            webpageList = [webpageTest]

            postCountermeasureOverhead = 0

            for w in webpageList:
                for trace in w.getTraces():
                    print 'ben/attck id: ' + str(trace.getId()) + '. trace id: ' + str(trace.getTraceIndex())
                    traceWithCountermeasure = trace
                    postCountermeasureOverhead += traceWithCountermeasure.getBandwidth()

            print 'num of syscalls: ' + str(postCountermeasureOverhead)
            print '--------------'



if __name__ == '__main__':
    unittest.main()
