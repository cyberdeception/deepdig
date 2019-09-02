# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import arffWriter
import wekaAPI
from glove import Glove
import numpy as np
from gensim import models

# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import arffWriter
import wekaAPI
from glove import Glove
import numpy as np
from gensim import models


class Word2VectClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance(trace):
        model = models.word2vec.Word2Vec.load("word2vecModel")
        return Word2VectClassifier.generateInstances(model,trace)

    @staticmethod
    def generateInstances(model,trace):
        myVectors = []
        vectorDict = {}
        paragraph = []
        for packet in trace.getPackets():
            key = str(packet.getLength()) + "_" + str(packet.getDirection())
            paragraph.append(key)

        directionCursor = None
        dataCursor      = 0
        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                dataKey = 'S'+str(directionCursor)+'-'+str( Word2VectClassifier.roundArbitrary(dataCursor, 600) )
                paragraph.append(dataKey)
                directionCursor = packet.getDirection()
                dataCursor      = 0

            dataCursor += packet.getLength()

        if dataCursor>0:
            key = 'S'+str(directionCursor)+'-'+str( Word2VectClassifier.roundArbitrary(dataCursor, 600) )
            paragraph.append(key)

        for key in paragraph:
            if key in model:
                myVectors.append(list(model[key]))
        # for each packet len get the vectors and sum it up by colum to get a 100 dim vector to represent a trace therefor an instance
        #myVectors = myglove.transform_paragraph(paragraph, epochs=90, ignore_missing=True)
        if len(myVectors) == 0:
            return None
        mymeanVector = np.mean(myVectors, axis=0)
        print mymeanVector.shape
        count = 0
        for l in mymeanVector:
            vectorDict["w" + str(count)] = l;
            count = count + 1;
        instance = {}  # trace.getHistogram()
        # print instance
        instance['class'] = 'webpage' + str(trace.getId())
        newinstances = dict(instance.items() + vectorDict.items())
        # some instances just contain nan values that should be discarded
        if np.isnan(vectorDict["w1"]):
            return None
        return newinstances


    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'])


























