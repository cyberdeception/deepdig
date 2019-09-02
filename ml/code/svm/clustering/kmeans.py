
#https://www.youtube.com/watch?v=ZS-IM9C3eFg
'''
from sklearn import cluster, datasets

iris = datasets.load_iris()
#print(iris)
X_iris = iris.data
print(X_iris)
y = iris.target
#print(y)

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
print(k_means.labels_[::10])

print k_means.cluster_centers_
'''

from Utils import Utils
import config
import os
import itertools
import shutil # to remove folders
import fnmatch
import numpy as np
import getopt
import sys
import classifiers.wekaAPI



def main():

    #Utils.testDelaunay()

    NumWebsites = 0
    numClusters = 0
    EigenVecLen = 10
    threshold = 10

    try:
        # m: number of websites
        # u: number of clusters
        # E: length of PCA components (Eigen Vectors)
        opts, args = getopt.getopt(sys.argv[1:], "m:u:E:h")
    except getopt.GetoptError, err:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o in ("-m"):
            NumWebsites = int(a)
        elif o in ("-u"):
            numClusters = int(a)
        elif o in ("-E"):
            EigenVecLen = int(a)
        else:
            sys.exit(2)

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/owC23C3_sameTraces_c0_A0/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/owC23C3_sameTraces_c0_A1/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingWithBug"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix/c0/m10"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix/c0/m40"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix/c0/m5"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix/c0/m60"

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/BugFixed/owC23C3_sameTraces_c0_A1/cache"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing/testingAfterBugFix/c0/m5/testingOneClassClustering/cache"

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/owC23C3_sameTraces_c0_A0/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/owC23C3_sameTraces_c8_A1/cache"
    #datafile-openworld5.91tjsedak300.c0.d0.C23.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-test.arff

    config.OUTPUT_DIR = os.path.join(FIELS_PATH, "outputOW")
    if not os.path.exists(config.OUTPUT_DIR):
        os.mkdir(config.OUTPUT_DIR)

    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)

                numMonitored = trainingFilename.split("openworld")[1]
                numMonitored = int(numMonitored.split(".")[0])

                fileID = trainingFilename.split("openworld")[1]
                fileID = fileID.split(".")[1]
                config.RUN_ID = fileID[:8]

                if numClusters == 0:
                    NumClusters = numMonitored
                else:
                    NumClusters = numClusters

                #Utils.calcWeightsLogisticRegressionTest()
                #points = np.random.rand(30, 2)
                #Utils.calcCvxHull(points)
                #Utils.calcCvxHull_Delaunay(points)
                if NumWebsites == 0:
                    print "\n\nkmeans:"
                    print "------------"
                    description = "kmeans"
                    Utils.calcKmeans([trainingFilename, testingFilename], numMonitored, NumClusters, description)

                    print "\n\ncalc PCA:"
                    config.n_components_PCA = EigenVecLen
                    [trainingFilenamePca,testingFilenamePca] = Utils.calcPCA_ow([trainingFilename,testingFilename])

                    print "\n\nkmeans with PCA " + str(config.n_components_PCA)
                    print "------------"
                    description = "kmeans with PCA " + str(config.n_components_PCA)
                    Utils.calcKmeans([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description)

                    #print "\n\nkmeans with PCA: " + str(config.n_components_PCA) + " + Convex Hull"
                    #print "------------"
                    #Utils.calcKmeansCvxHullDelaunay([trainingFilenamePca,testingFilenamePca], numMonitored, numClusters, description)
                    #Utils.calcKmeansCvxHullDelaunay_Testing([trainingFilenamePca,testingFilenamePca], numMonitored)

                    print "\n\nkmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed"
                    print "------------"
                    description = "kmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed"
                    Utils.calcKmeansCvxHullDelaunay_Mixed([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description)



                    print "\n\nkmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed -- KNN"
                    print "------------"

                    description = "kmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed -- KNN (threshold = " + str(threshold) + ")"
                    Utils.calcKmeansCvxHullDelaunay_Mixed_KNN([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description,  threshold)

                else:
                    # check file if file numMonitored matches the passed one
                    # datafile-openworld5.i2vt8sjxk300.c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff
                    arffFileName = trainingFilename.split("datafile-openworld")[1] # emzxqu17k30.c0.d0.C23.N775.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0.P0.G0.l0.25.b600-train.arff

                    numMonitored = int(arffFileName.split(".")[0]) #

                    if NumWebsites == numMonitored:
                        print "\n\nkmeans::"
                        print "------------"
                        description = "kmeans"
                        Utils.calcKmeans([trainingFilename, testingFilename], numMonitored, NumClusters, description)

                        print "\n\ncalc PCA:"
                        config.n_components_PCA = EigenVecLen
                        [trainingFilenamePca,testingFilenamePca] = Utils.calcPCA_ow([trainingFilename,testingFilename])

                        print "\n\nkmeans with PCA " + str(config.n_components_PCA)
                        print "------------"
                        description = "kmeans with PCA " + str(config.n_components_PCA)
                        Utils.calcKmeans([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description)

                        #print "\n\nkmeans with PCA: " + str(config.n_components_PCA) + " + Convex Hull"
                        #print "------------"
                        #Utils.calcKmeansCvxHullDelaunay([trainingFilenamePca,testingFilenamePca], numMonitored, numClusters, description)
                        #Utils.calcKmeansCvxHullDelaunay_Testing([trainingFilenamePca,testingFilenamePca], numMonitored)

                        print "\n\nkmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed"
                        print "------------"
                        description = "kmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed"
                        Utils.calcKmeansCvxHullDelaunay_Mixed([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description)



                        print "\n\nkmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed -- KNN"
                        print "------------"

                        description = "kmeans with PCA " + str(config.n_components_PCA) + " + Convex Hull -- Mixed -- KNN (threshold = " + str(threshold) + ")"
                        Utils.calcKmeansCvxHullDelaunay_Mixed_KNN([trainingFilenamePca,testingFilenamePca], numMonitored, NumClusters, description,  threshold)



if __name__ == "__main__":
    main()












































