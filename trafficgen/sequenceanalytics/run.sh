#!/bin/bash
python extractSeriesForTree.py 1 90 500 att
python entropyCalcTree.py att
python featselect2Tree.py 1_90_500_att


python extractSeriesForTree.py 2 70 200 att
python entropyCalcTree.py att
python featselect2Tree.py 2_70_200_att

python extractSeriesForTree.py 2 70 200 ben
python entropyCalcTree.py ben
python featselect2Tree.py 2_70_200_ben


python extractSeriesForTree.py 1 70 200 att
python entropyCalcTree.py att
python featselect2Tree.py 1_70_200_att

python extractSeriesForTree.py 1 70 200 ben
python entropyCalcTree.py ben
python featselect2Tree.py 1_70_200_ben


python extractSeriesForTree.py 0 70 200 att
python entropyCalcTree.py att       
python featselect2Tree.py 0_70_200_att

python extractSeriesForTree.py 0 70 200 ben
python entropyCalcTree.py ben
python featselect2Tree.py 0_70_200_ben


python extractSeriesForTree.py 2 70 300 att
python entropyCalcTree.py att
python featselect2Tree.py 2_70_300_att

python extractSeriesForTree.py 2 70 300 ben
python entropyCalcTree.py ben
python featselect2Tree.py 2_70_300_ben


python extractSeriesForTree.py 0 70 300 att
python entropyCalcTree.py att
python featselect2Tree.py 0_70_300_att

python extractSeriesForTree.py 2 70 300 ben
python entropyCalcTree.py ben
python featselect2Tree.py 2_70_300_ben




python extractSeriesForTree.py 1 70 300 att
python entropyCalcTree.py att
python featselect2Tree.py 1_70_300_att

python extractSeriesForTree.py 1 70 300 ben
python entropyCalcTree.py ben
python featselect2Tree.py 1_70_300_ben


python extractSeriesForTreebin.py 1 70 300 
python entropyCalcTreebin.py 
python featselect2Tree.py 1_70_300_bin

python extractSeriesForTreebin.py 0 70 300
python entropyCalcTreebin.py
python featselect2Tree.py 0_70_300_bin

