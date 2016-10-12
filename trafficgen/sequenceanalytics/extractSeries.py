import os

import sys


dataset = [1,2,3,4,5,6,7,8,9,10,11,12,20]
for i in dataset:
    os.system("rm ./outfile"+str(i))
os.system("rm ./index_map")

def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line)

def getArg(instruct,inText):
    myInst = ["read","write","fcntl","close","open","fstat"]
    if instruct in myInst:
       startIndex = inText.find('>')
       endIndex = inText.find(')')
       if startIndex != -1 and endIndex != -1 :
          return inText[startIndex+1:endIndex]
   
   
#read,write,fcntl,close,open,fstat
def getSet():
    ops = set()
    for i in dataset:
      for j in range(100,150):
        filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
        with open(filename, "r") as f:
             for line in f:
                 line = line.rstrip()
                 entry = line.split(" ")
                 if entry[4] != "tcpdump":
                    ops.add(entry[6])
                    if len(entry) >= 8:
                       myarg = getArg(entry[6],entry[7])
                       if myarg != None:
                          ops.add(myarg)
                       

    return ops


inst = list(getSet())
print inst

for j in range(0,len(inst)):
    print str(j), inst[j]
    if inst[j] == '':
       inst[j] = 'null'
    line = ""+str(j)+" " + inst[j]
    appendToFile("./index_map",line+"\n")


def getIndex(instname):
    for i in range(0,len(inst)):
        if inst[i] == instname:
           return i
    return 0;
    


for i in dataset:
    for j in range(100,200):
        instance = []
        filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
        with open(filename, "r") as f:
             for line in f:
                 line = line.rstrip()
                 entry = line.split(" ")
                 if entry[4] != "tcpdump":
                    if sys.argv[1] == '1' or sys.argv[1] == '2':#only instruction
                       if entry[6] != '':
                          instance.append(str(getIndex(entry[6])))
                    if len(entry) >= 8:
                       myarg = getArg(entry[6],entry[7])
                       if myarg != None:
                          if sys.argv[1] == '0' or sys.argv[1] == '2':#only arg
                             if myarg != '':
                                instance.append(str(getIndex(myarg)))
        line_instance = " ".join(list(set(instance[0:400])))
        appendToFile("./outfile"+str(i),line_instance+ " -1 ")
	appendToFile("./outfile"+str(i),"-2\n")

for i in dataset:
    #os.system("java -Xmx2g -jar spmf.jar run TKS "+ "./outfile"+str(i)+" output"+str(i)+ " 100")               
    os.system("java -Xmx2g -jar spmf.jar run CM-SPAM "+ "./outfile"+str(i)+" output"+str(i)+ " " +sys.argv[2]+ "% ")



