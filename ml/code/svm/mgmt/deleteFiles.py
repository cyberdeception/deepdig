
import config
import os
import itertools
import shutil # to remove folders
import fnmatch
import numpy as np
import getopt
import sys

def main():

    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/testing"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/owC23C3_sameTraces_c0_A0/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/BugFixed/owC23C3_sameTraces_c0_A1/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/BugFixed/owC23C3_sameTraces_c1_A1/cache"
    #FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/BugFixed/owC23C3_sameTraces_c8_A1/cache"
    FIELS_PATH="/data/kld/papers/0SubmittedPapers/WF_icde16/VM_Local_experiments/owC23C3/BugFixed/owC23C3_sameTraces_c9_A1/cache"
    #datafile-openworld5.91tjsedak300.c0.d0.C3.N775.t40.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train-orig.arff
    for (path, dirs, files) in os.walk(FIELS_PATH):
        for myfile in files:
            if fnmatch.fnmatch(myfile, '*train-orig.arff'):
                toBeDelFile = os.path.join(FIELS_PATH, myfile)
                if os.path.exists(toBeDelFile):
                    print toBeDelFile + " to be deleted!"
                    os.remove(toBeDelFile)




if __name__ == "__main__":
    main()