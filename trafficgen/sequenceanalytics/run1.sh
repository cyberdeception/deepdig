#!/bin/bash


python extractSeriesForTree.py 1 70 20000 att
python entropyCalcTree.py att
python featselect2Tree.py 1_70_20000_att

#python extractSeriesForTree.py 2 70 20000 att
#python entropyCalcTree.py att
#python featselect2Tree.py 2_70_20000_att

#python extractSeriesForTree.py 0 70 20000 att
#python entropyCalcTree.py att
#python featselect2Tree.py 0_70_20000_att


#python extractSeriesForTree.py 1 70 20000 ben
#python entropyCalcTree.py ben
#python featselect2Tree.py 1_70_20000_ben

#python extractSeriesForTree.py 0 70 20000 ben
#python entropyCalcTree.py ben
#python featselect2Tree.py 0_70_20000_ben

