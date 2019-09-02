
import config
from Utils import Utils
import itertools
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import numpy as np


def calcKmeans(files, numClusters=-1,description=""):

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    trainingInstancesList = []
    clusterClasses = []

    X_testing=[]
    Y_testing=[]
    '''
    for line in trainList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                monClasses = line.split(" ")[2].split("{")[1].split("}")[0].split(",")
        else:
            #instancesList.append(float(line.split(",")[:-1]))
            trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
            #y.append(line.split(",")[-1])
    '''
    for line in trainList:
        if line[0] != '@':
            trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
            currY = line.split(",")[-1]
            if not clusterClasses.__contains__(currY):
                clusterClasses.append(currY)

    if (numClusters==-1):
        numClusters = len(clusterClasses)

    '''
    for line in testList:
        if line[0] != '@':
             currY = line.split(",")[-1]
             if monClasses.__contains__(currY):  # add all testing monitored instances
                 X_testing.append([float(i) for i in line.split(",")[:-1]])
                 Y_testing.append(line.split(",")[-1])
             else: # nonMonitored instance
                 if not unmonClasses.__contains__(currY): # add one instance only from unmonitored classes
                     unmonClasses.append(currY)
                     X_testing.append([float(i) for i in line.split(",")[:-1]])
                     Y_testing.append(currY)
    '''
    for line in testList:
        if line[0] != '@':
            X_testing.append([float(i) for i in line.split(",")[:-1]])
            Y_testing.append(line.split(",")[-1])

        #if line[0] != '@':
        #    X_testing.append([float(i) for i in line.split(",")[:-1]])
        #    Y_testing.append(line.split(",")[-1])

    #print instancesList

    X = np.array(trainingInstancesList)

    #X = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])
    #print X

    #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA

    #km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1, verbose=0)
    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1, verbose=0)

    km.fit(X) # building the clusters from the monitored instances only

    #print km.cluster_centers_[0]

    #print km.labels_

    # indexes of point in a specific cluster
    #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

    #print index

    # get radius of each cluster
    radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros
    #print radius
    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        maxDist = -1
        for i in pointsIndex:
            # Euclidean distance
            currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
            if currDist > maxDist:
                radius[clusIndx] = currDist
            #maxDist = currDist
                maxDist = currDist



    #X_testing = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])

    #Y_testing = np.array([0,
    #              0,
    #              0,
    #              1,
    #              1,
    #              1])

    #monClasses = [0,1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    inside = False

    for i in range(len(X_testing)):
        inside = False
        for clusIndx in range(len(km.cluster_centers_)):
           dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
           if dist <= radius[clusIndx]: #/1.5:#/2.0:
               inside = True
        '''
        if inside:
           if clusterClasses.__contains__(Y_testing[i]):
               tp += 1
           else:
               fp += 1
        else:
           if clusterClasses.__contains__(Y_testing[i]):
               fn += 1
           else:

               tn += 1
        '''
        if inside:
           if clusterClasses.__contains__(Y_testing[i]):
               tn += 1
           else:
               fn += 1
        else:
           if clusterClasses.__contains__(Y_testing[i]):
               fp += 1
           else:
               tp += 1

    print "\n"
    print "radii: "
    print radius
    print "NumClusters: " + str(numClusters)
    print "dataset: " + str(files)

    print "tp = " + str(tp)
    print "tn = " + str(tn)
    print "fp = " + str(fp)
    print "fn = " + str(fn)

    tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
    fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
    Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    print "tpr = " + tpr
    print "fpr = " + fpr
    print "Acc = " + Acc
    print "F2  = " + F2

    output = []
    output.append(tpr)
    output.append(fpr)
    output.append(Acc)
    output.append(F2)
    output.append(str(tp))
    output.append(str(tn))
    output.append(str(fp))
    output.append(str(fn))
    output.append(description)
    output.append(numClusters)
    output.append(config.RUN_ID)


    summary = '\t, '.join(itertools.imap(str, output))

    outputFilename = Utils.getOutputFileName(files[0])

    #f = open( outputFilename+'.output', 'a' )
    f = open( outputFilename, 'a' )
    f.write( "\n"+summary )
    f.close()

    print ''



#############################################################################


def calcKmeansCvxHullDelaunay(files, numClusters=-1,description=""):

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    trainingInstancesList = []
    clusterClasses = []

    X_testing=[]
    Y_testing=[]

    for line in trainList:
        if line[0] != '@':
            trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
            currY = line.split(",")[-1]
            if not clusterClasses.__contains__(currY):
                clusterClasses.append(currY)

    if (numClusters==-1):
        numClusters = len(clusterClasses)

    for line in testList:
        if line[0] != '@':
            X_testing.append([float(i) for i in line.split(",")[:-1]])
            Y_testing.append(line.split(",")[-1])


    X = np.array(trainingInstancesList)

    #X = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])
    #print X

    # preprocessing, normalizing
    #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before CPA
    #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before CPA

    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
            verbose=0)

    km.fit(X) # building the clusters from the monitored instances only

    #print km.cluster_centers_[0]

    #print km.labels_

    # indexes of point in a specific cluster
    #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

    #print index

    # get radius of each cluster
    radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

    hull = []

    #print radius
    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
        #hull.append(Delaunay(clusterPoints))

        # hull needs at least 12 points
        if (len(clusterPoints)>12):
            hull.append(Delaunay(clusterPoints))


    print '#hulls: ' + str(len(hull))
    #X_testing = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])

    #Y_testing = np.array([0,
    #              0,
    #              0,
    #              1,
    #              1,
    #              1])

    #monClasses = [0,1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    inside = False

    for i in range(len(X_testing)):
        inside = False
        for hullIndx in range(len(hull)):
           if Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
               inside = True


        if inside:
           if clusterClasses.__contains__(Y_testing[i]):
               tn += 1
           else:
               fn += 1
        else:
           if clusterClasses.__contains__(Y_testing[i]):
               fp += 1
           else:
               tp += 1

    print "\n"
    print "radii: "
    print radius
    print "NumClusters: " + str(numClusters)
    print "dataset: " + str(files)

    print "tp = " + str(tp)
    print "tn = " + str(tn)
    print "fp = " + str(fp)
    print "fn = " + str(fn)

    tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
    fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
    Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    print "tpr = " + tpr
    print "fpr = " + fpr
    print "Acc = " + Acc
    print "F2  = " + F2

    output = []
    output.append(tpr)
    output.append(fpr)
    output.append(Acc)
    output.append(F2)
    output.append(str(tp))
    output.append(str(tn))
    output.append(str(fp))
    output.append(str(fn))
    output.append(description)
    output.append(numClusters)


    summary = '\t, '.join(itertools.imap(str, output))

    outputFilename = Utils.getOutputFileName(files[0])

    f = open( outputFilename, 'a' )
    f.write( "\n"+summary )
    f.close()

    print ''


###########################################################################


def calcKmeansCvxHullDelaunay_Mixed(files, numClusters=-1,description=""):

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    trainingInstancesList = []
    clusterClasses = []

    X_testing=[]
    Y_testing=[]

    for line in trainList:
        if line[0] != '@':
            trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
            currY = line.split(",")[-1]
            if not clusterClasses.__contains__(currY):
                clusterClasses.append(currY)

    if (numClusters==-1):
        numClusters = len(clusterClasses)

    for line in testList:
        if line[0] != '@':
            X_testing.append([float(i) for i in line.split(",")[:-1]])
            Y_testing.append(line.split(",")[-1])


    X = np.array(trainingInstancesList)

    #X = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])
    #print X

    # preprocessing, normalizing
    #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before PCA
    #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before PCA

    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
            verbose=0)

    km.fit(X) # building the clusters from the monitored instances only

    #print km.cluster_centers_[0]

    #print km.labels_

    # indexes of point in a specific cluster
    #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

    #print index

    # get radius of each cluster
    radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        maxDist = -1
        for i in pointsIndex:
            # Euclidean distance
            currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
            if currDist > maxDist:
                radius[clusIndx] = currDist
            #maxDist = currDist
                maxDist = currDist



    hull = []

    fewPointsClusters = [] # indexes of clusters where there are < 12 points (convex hull needs 12 points)

    #print radius
    #http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.Delaunay.html
    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
        if len(clusterPoints) >= 12:
            try:
                hull.append(Delaunay(clusterPoints))#,qhull_options="C-0"))
            except:
                print "Convex Hull ERROR"
                description += "Convex Hull ERROR"
                print " Cluster # " + str(clusIndx) + " -- Convex Hull ERROR. Kmeans cluster is to be checked for the participating points."
                fewPointsClusters.append(clusIndx)
                pass
        else:
            print " Cluster # " + str(clusIndx) + " doesn't have enough points to build a hull. Kmeans cluster is to be checked for the participating points."
            fewPointsClusters.append(clusIndx)


    #X_testing = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])

    #Y_testing = np.array([0,
    #              0,
    #              0,
    #              1,
    #              1,
    #              1])

    #monClasses = [0,1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    inside = False

    #looping over mixed (hulls + not-enough-point clusters)
    for i in range(len(X_testing)):
        inside = False

        # looping over hulls
        for hullIndx in range(len(hull)):
            if Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                inside = True

        # looping over not-enough-point clusters
        for clusIndx in range(len(km.cluster_centers_)):
            if fewPointsClusters.__contains__(clusIndx):
                #print " Cluster # " + str(clusIndx) + " is being examined."
                dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
                if dist <= radius[clusIndx]: #/1.5:#/2.0:
                    inside = True

        if inside:
           if clusterClasses.__contains__(Y_testing[i]):
               tn += 1
           else:
               fn += 1
        else:
           if clusterClasses.__contains__(Y_testing[i]):
               fp += 1
           else:
               tp += 1


    print "\n"
    print "radii: "
    print radius
    print "NumClusters: " + str(numClusters)
    print "dataset: " + str(files)

    print "tp = " + str(tp)
    print "tn = " + str(tn)
    print "fp = " + str(fp)
    print "fn = " + str(fn)

    tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
    fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
    Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    print "tpr = " + tpr
    print "fpr = " + fpr
    print "Acc = " + Acc
    print "F2  = " + F2

    output = []
    output.append(tpr)
    output.append(fpr)
    output.append(Acc)
    output.append(F2)
    output.append(str(tp))
    output.append(str(tn))
    output.append(str(fp))
    output.append(str(fn))
    output.append(description)
    output.append(numClusters)
    output.append(config.RUN_ID)


    summary = '\t, '.join(itertools.imap(str, output))

    outputFilename = Utils.getOutputFileName(files[0])

    f = open( outputFilename, 'a' )
    f.write( "\n"+summary )
    f.close()

    print ''


###########################################################################


def calcKmeansCvxHullDelaunay_Mixed_KNN(files, numClusters=-1,description="", threshold=3):

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    trainingInstancesList = []
    clusterClasses = []

    X_testing=[]
    Y_testing=[]

    for line in trainList:
        if line[0] != '@':
            trainingInstancesList.append([float(i) for i in line.split(",")[:-1]])
            currY = line.split(",")[-1]
            if not clusterClasses.__contains__(currY):
                clusterClasses.append(currY)

    if (numClusters==-1):
        numClusters = len(clusterClasses)

    for line in testList:
        if line[0] != '@':
            X_testing.append([float(i) for i in line.split(",")[:-1]])
            Y_testing.append(line.split(",")[-1])


    X = np.array(trainingInstancesList)

    #X = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])
    #print X

    # preprocessing, normalizing
    #X = (X - np.mean(X, 0)) / np.std(X, 0) # scale data before PCA
    #X_testing = (X_testing - np.mean(X_testing, 0)) / np.std(X_testing, 0) # scale data before PCA

    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=1,
            verbose=0)

    km.fit(X) # building the clusters from the monitored instances only

    #print km.cluster_centers_[0]

    #print km.labels_

    # indexes of point in a specific cluster
    #index = [x[0] for x, value in np.ndenumerate(km.labels_) if value==0] # value==cluster number

    #print index

    # get radius of each cluster
    radius = [0]*len(km.cluster_centers_) # initialize the radius list to zeros

    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        maxDist = -1
        for i in pointsIndex:
            # Euclidean distance
            currDist = np.linalg.norm(X[i] - km.cluster_centers_[clusIndx])
            if currDist > maxDist:
                radius[clusIndx] = currDist
            #maxDist = currDist
                maxDist = currDist



    hull = []

    fewPointsClusters = [] # indexes of clusters where there are < 12 points (convex hull needs 12 points)

    #print radius
    #http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.Delaunay.html
    for clusIndx in range(len(km.cluster_centers_)):
        # indexes of points in a specific cluster
        pointsIndex = [x[0] for x, value in np.ndenumerate(km.labels_) if value==clusIndx]
        clusterPoints = list(X[pointsIndex]) # Access multiple elements of list (here X) knowing their index
        if len(clusterPoints) >= 12:
            try:
                hull.append(Delaunay(clusterPoints))#,qhull_options="C-0"))
            except:
                print "Convex Hull ERROR"
                description += " Convex Hull ERROR"
                print " Cluster # " + str(clusIndx) + " -- Convex Hull ERROR. Kmeans cluster is to be checked for the participating points."
                fewPointsClusters.append(clusIndx)
                pass
        else:
            print " Cluster # " + str(clusIndx) + " doesn't have enough points to build a hull. Kmeans cluster is to be checked for the participating points."
            fewPointsClusters.append(clusIndx)


    #X_testing = np.array([[1, 2],
    #              [5, 8],
    #              [1.5, 1.8],
    #              [8, 8],
    #              [1, 0.6],
    #              [9, 11]])

    #Y_testing = np.array([0,
    #              0,
    #              0,
    #              1,
    #              1,
    #              1])

    #monClasses = [0,1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    inside = False

    #looping over mixed (hulls + not-enough-point clusters)
    for i in range(len(X_testing)):

        inside = False

        # looping over hulls
        for hullIndx in range(len(hull)):
            if inside != True and Utils.in_hull(X_testing[i],hull[hullIndx]): # returns true if point is inside hull
                inside = True

            # KNN
            if inside != True and Utils.is_knn_to_hull_border_points(X, X_testing[i],hull[hullIndx],threshold):
                inside = True

        # looping over not-enough-point clusters
        if inside != True:
            for clusIndx in range(len(km.cluster_centers_)):
                if fewPointsClusters.__contains__(clusIndx):
                    #print " Cluster # " + str(clusIndx) + " is being examined."
                    dist = np.linalg.norm(X_testing[i] - km.cluster_centers_[clusIndx])
                    if dist <= radius[clusIndx]: #/1.5:#/2.0:
                        inside = True

        if inside:
           if clusterClasses.__contains__(Y_testing[i]):
               tn += 1
           else:
               fn += 1
        else:
           if clusterClasses.__contains__(Y_testing[i]):
               fp += 1
           else:
               tp += 1


    print "\n"
    print "radii: "
    print radius
    print "NumClusters: " + str(numClusters)
    print "dataset: " + str(files)

    print "tp = " + str(tp)
    print "tn = " + str(tn)
    print "fp = " + str(fp)
    print "fn = " + str(fn)

    tpr = str( "%.2f" % (float(tp)/float(tp+fn)) )
    fpr = str( "%.2f" % (float(fp)/float(fp+tn) ))
    Acc = str( "%.2f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F2  = str( "%.2f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    print "tpr = " + tpr
    print "fpr = " + fpr
    print "Acc = " + Acc
    print "F2  = " + F2

    output = []
    output.append(tpr)
    output.append(fpr)
    output.append(Acc)
    output.append(F2)
    output.append(str(tp))
    output.append(str(tn))
    output.append(str(fp))
    output.append(str(fn))
    output.append(description)
    output.append(numClusters)
    output.append(config.RUN_ID)


    summary = '\t, '.join(itertools.imap(str, output))

    outputFilename = Utils.getOutputFileName(files[0])

    f = open( outputFilename, 'a' )
    f.write( "\n"+summary )
    f.close()

    print ''


###########################################################################