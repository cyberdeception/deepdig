
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing.pool import ThreadPool
# KNN algorithm with Longest Common Subsequence as the distance metric

def lcsthread(neigh,data):
  prediction = neigh.predict(np.arange(len(data)).reshape(-1, 1))
  return prediction

class KNN_LCS:

    def __init__(self, Xtrain, Xtest, Ytrain, Ytest, neighbors):
        self.__Xtrain = Xtrain
        self.__Xtest = Xtest
        self.__Ytrain = Ytrain
        self.__Ytest = Ytest
        self.__neighbors = neighbors
        self.__Yprediction = []

    def getXTrain(self): return self.__Xtrain
    def getXTest(self): return self.__Xtest
    def getYTrain(self): return self.__Ytrain
    def getYTest(self): return self.__Ytest
    def getNeighbors(self): return self.__neighbors

    def setXTrain(self,Xtrain): self.__Xtrain = Xtrain
    def setXTest(self,Xtest): self.__Xtest = Xtest
    def setYTrain(self,Ytrain): self.__Ytrain = Ytrain
    def setYTest(self,Ytest): self.__Ytest = Ytest
    def setNeighbors(self,neighbors): self.__neighbors = neighbors

    def calcHP_KNN_LCS(self):

        XX = np.arange(len(self.__Xtrain)).reshape(-1, 1) # an index into a separate data structure (Xtrain), to overcome different dimensions for each trace
        y = np.array(self.__Ytrain)

        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=self.__neighbors, metric=self.LCS_dist, algorithm='brute') # kNN distance metric overwritten, see LCS_dist. algorithm='brute' is a must

        neigh.fit(XX, y)

         # an index into a separate data structure (Xtest).
        pool = ThreadPool(processes=5)

        numThreads = 10
        stepSize = len(self.__Xtest) / numThreads
        async_result = [None] * numThreads
        for i in range(0,numThreads):
            async_result[i] =  pool.apply_async(lcsthread, (neigh,self.__Xtest[stepSize*i:stepSize*(i+1)]))

# do some other stuff in the main process
        for i in range(0,numThreads):
            self.__Yprediction[stepSize*i:stepSize*(i+1)] = async_result[i].get()


    def LCS_dist(self,xx, X): # xx is the testing instance, X is the training instance (passed one by one)
        import mlpy

        i, j = int(xx[0]), int(X[0])     # extract indices, i for testing, j for training

        #print "i = " + str(i) + ", j = " +str(j)

        length, path = mlpy.lcs_std(self.__Xtest[i], self.__Xtrain[j])

        dist_lcs = float(length)/np.sqrt(len(self.__Xtest[i])*len(self.__Xtrain[j])) ## Formula taken from section 4.1.2 in paper: Anomaly Detection for Discrete Sequences: A Survey

        if dist_lcs != 0:
            dist_lcs_inv = float(1/float(dist_lcs))
        else:
            dist_lcs_inv = 9999.0

        return dist_lcs_inv


    def getAccuracyDebugInfo(self):

        self.calcHP_KNN_LCS()

        totalPredictions = 0
        totalCorrectPredictions = 0
        debugInfo = []

        for i in range(len(self.__Ytest)):
            totalPredictions += 1.0
            actualClass = self.__Ytest[i]
            predictedClass = self.__Yprediction[i]
            debugInfo.append([actualClass,predictedClass])
            if actualClass == predictedClass:
                totalCorrectPredictions += 1.0


        accuracy = totalCorrectPredictions / totalPredictions * 100.0

        return [accuracy,debugInfo]


''' testing
#Xtrain = np.array([[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]])
#Xtest = np.array([[2,1,6,5,4,4],[2,1,6,5,4]])

Xtrain = [[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]]
Xtest = [[2,1,6,5,4,4],[2,1,6,5,4],[2,1,5,4,4],[2,4]]

#Ytrain =  np.array([[0], [0], [0], [1], [1], [1]])
#Ytrain =  [0, 0, 0, 1, 1, 1]
Ytrain =  ['webpage0', 'webpage0', 'webpage0', 'webpage1', 'webpage1', 'webpage1']
#Ytest =  np.array([[1], [0]])
#Ytest =  ['webpage0', 'webpage1']
Ytest =  ['webpage0', 'webpage1', 'webpage0', 'webpage1']
knn_lcs_obj = KNN_LCS(Xtrain, Xtest, Ytrain, Ytest, neighbors=3)

#knn_lcs_obj.calcHP_KNN_LCS()
print knn_lcs_obj.getAccuracyDebugInfo()
'''
'''
Xtrain = [[1,2,6,5,4,8], [2,1,6,5,4,4], [2,1,6,5], [2,1,6,5,4,4,7,6], [2,1,6,5,4],[2,3,6,5,4,4]]
Xtest = [[2,1,6,5,4,4],[2,1,6,5,4],[2,1,5,4,4],[2,4]]
Ytrain =  ['webpage0', 'webpage0', 'webpage0', 'webpage1', 'webpage1', 'webpage1']
Ytest =  ['webpage0', 'webpage1', 'webpage0', 'webpage1']
knn_lcs_obj = KNN_LCS(Xtrain, Xtest, Ytrain, Ytest, neighbors=3)
print knn_lcs_obj.getAccuracyDebugInfo()

'''

