trpath="/workspace/datafiles_oml/ngramnormaltrain.csv"
testpath="/workspace/datafiles_oml/ngramnormaltest.csv"


python3 mainalltest.py --iterations 100 --trainpath $trpath --testpath $testpath --classes "3,4,5,6,7,8,9,10,11,12,14,16,17,22,23,24,25,26,27,28,29,30,31,32,33,34,35"




trpath="/workspace/datafiles_oml/packetnormaltrain.csv"
testpath="/workspace/datafiles_oml/packetnormaltest.csv"


python3 mainalltest.py --iterations 100 --trainpath $trpath --testpath $testpath --classes "3,4,5,6,7,8,9,10,11,12,14,16,17,22,23,24,25,26,27,28,29,30,31,32,33,34,35"

