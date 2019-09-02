import os
import fnmatch

numAttacks=17 # 16  attacks and no attack -w 0

#TPR
TPR = [None] * numAttacks

for i in range(len(TPR)):
    TPR[i] = [None] * 4 # num, C23, C43, ensemble

#FPR
FPR = [None] * numAttacks

for i in range(len(FPR)):
    FPR[i] = [None] * 4

#ACC
ACC = [None] * numAttacks

for i in range(len(ACC)):
    ACC[i] = [None] * 4

#F2
F2 = [None] * numAttacks

for i in range(len(F2)):
    F2[i] = [None] * 4


desc = os.path.dirname(os.path.realpath(__file__)).split(os.sep)[-2]

def __writeFile(resultsList, type):
    resultType = type
    dir = os.path.join('.', desc)
    if not os.path.exists(dir):
        os.mkdir(dir)

    f = open( os.path.join(dir, resultType), 'w' )
    f.write("#"+desc+"\n")
    for entry in resultsList:
        f.write( str(entry[0])+'\t'+str(entry[1])+'\t'+str(entry[2])+'\t'+str(entry[3])+"\n" )

    f.close()


def calcAvgs(lines):
    numExperiments = lines.__len__()
    tprSum = 0.0
    fprSum = 0.0
    accSum = 0.0
    f2Sum = 0.0

    for line in lines:
        tprSum += float(line[0])
        fprSum += float(line[1])
        accSum += float(line[2])
        f2Sum += float(line[3])

    tprAvg = tprSum / numExperiments
    fprAvg = fprSum / numExperiments
    accAvg = accSum / numExperiments
    f2Avg = f2Sum / numExperiments

    return [tprAvg, fprAvg, accAvg, f2Avg]



for (path, dirs, files) in os.walk('.'):
    for myfile in files:
        if fnmatch.fnmatch(myfile, '*binary'):
            fileLines = [line.strip() for line in open(os.path.join('.', myfile))]

            linesResult = []
            for fileLine in fileLines:
                if not fileLine.startswith("tpr"):
                    linesResult.append(fileLine.split(",")) # [tpr	, fpr	, Acc	, F2	, tp	, tn	, fp	, fn	, File ID]
                    # was
                    # lineResults = fileLine.split(",") # [tpr	, fpr	, Acc	, F2	, tp	, tn	, fp	, fn	, File ID]

            lineResultsAvg = calcAvgs(linesResult)

            C = int(myfile.split(".C")[1].split(".")[0])
            w = int(myfile.split(".w")[1].split(".")[0])

            TPR[w][0] = w
            FPR[w][0] = w
            ACC[w][0] = w
            F2[w][0]  = w

            if C == 23 and "ensemble" not in myfile:
                TPR[w][1] = '%.2f' % (float(lineResultsAvg[0]) * 100)
                FPR[w][1] = '%.2f' % (float(lineResultsAvg[1]) * 100)
                ACC[w][1] = '%.2f' % (float(lineResultsAvg[2]) * 100)
                F2[w][1]  = '%.2f' % (float(lineResultsAvg[3]) * 100)
            elif C == 43 and "ensemble" not in myfile:
                TPR[w][2] = '%.2f' % (float(lineResultsAvg[0]) * 100)
                FPR[w][2] = '%.2f' % (float(lineResultsAvg[1]) * 100)
                ACC[w][2] = '%.2f' % (float(lineResultsAvg[2]) * 100)
                F2[w][2]  = '%.2f' % (float(lineResultsAvg[3]) * 100)
            elif C == 43 and "ensemble" in myfile:
                TPR[w][3] = '%.2f' % (float(lineResultsAvg[0]) * 100)
                FPR[w][3] = '%.2f' % (float(lineResultsAvg[1]) * 100)
                ACC[w][3] = '%.2f' % (float(lineResultsAvg[2]) * 100)
                F2[w][3]  = '%.2f' % (float(lineResultsAvg[3]) * 100)
            else:
                print "Code shouldn't come to here!"



print 'tpr'
for entry in TPR:
    print entry
    print '\n'
print 'fpr'
for entry in FPR:
    print entry
    print '\n'

__writeFile(TPR, 'tpr')
__writeFile(FPR, 'fpr')
__writeFile(ACC, 'acc')
__writeFile(F2,  'f2')


