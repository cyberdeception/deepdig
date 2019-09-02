trpath="/workspace/datafiles_oml/ngramhumantrain.csv"
testpath="/workspace/datafiles_oml/ngramhumantest.csv"


python3 mainalltest.py --iterations 100 --trainpath $trpath --testpath $testpath --classes "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37"


trpath="/workspace/datafiles_oml/packethumantrain.csv"
testpath="/workspace/datafiles_oml/packethumantest.csv"


python3 mainalltest.py --iterations 100 --trainpath $trpath --testpath $testpath --classes "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37"

