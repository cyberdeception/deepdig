
import config
import os
import itertools
from Utils import Utils
import shutil # to remove folders
import fnmatch
import numpy as np
import wekaAPI
import getopt
import sys

def main():

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/WebsiteFingerprinting1/processing/upsupervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/processing/supervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/processing/supervised"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2Processing/datasets"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/FeatureSelection/Try2MM/pythonCode/cache"
    #datafile-fiqtk29xk20.c0.d0.C23.N755.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A0.V0-train.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train.arff'):
                trainingFilename = os.path.join(FIELS_PATH, myfile)
                testingFilename = myfile[:-10] + "test.arff"
                testingFilename = os.path.join(FIELS_PATH, testingFilename)

                featuresFilename = myfile[:-10] + "features.arff"
                featuresFilename = os.path.join(FIELS_PATH, featuresFilename)

                if not os.path.exists(featuresFilename):
                    f = open( featuresFilename, 'w' )
                    f.close()

                featuresFilename_2 = myfile[:-10] + "features_2.arff"
                featuresFilename_2 = os.path.join(FIELS_PATH, featuresFilename_2)

                if not os.path.exists(featuresFilename_2):
                    f = open( featuresFilename_2, 'w' )
                    f.close()



if __name__ == "__main__":
    main()