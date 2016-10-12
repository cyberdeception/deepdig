import os

import sys

numinstance = 10
if sys.argv[4] == "att":
   dataset = [1,2,3,4,5,6,7,8,9,10,11,12]
elif sys.argv[4] == "ben":
   dataset = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25] #13 20 benstarts from 14

for i in dataset:
    for k in range(0,numinstance):
        os.system("rm ./outfile"+str(i)+"_"+str(k))
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
      if i < 14:
         startindex = 100
         endindex = 200
      else:
         startindex = 0
         endindex = 100
      for j in range(startindex,endindex):
        filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
        with open(filename, "r") as f:
             linecounter = 0
             for line in f:
                 #if linecounter > int(sys.argv[3]):
                    #break 
                 line = line.rstrip()
                 entry = line.split(" ")
                 if entry[4] != "tcpdump":
                    linecounter = linecounter +1
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
    for k in range(0,numinstance):
            if i < 14:
               startindex = 100
               endindex = 200
            else:
               startindex = 0
               endindex = 100
	    for j in range(startindex+ 10*k,startindex+ 10*k + 10):
            #for j in range(startindex+ 1*k,startindex+ 1*k + 1):#1 file per seq 
		instance = []
		filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
		with open(filename, "r") as f:
                     linecounter = 0
		     for line in f:
                         #if linecounter > int(sys.argv[3]):
                            #break 
		         line = line.rstrip()
		         entry = line.split(" ")
		         if entry[4] != "tcpdump":
                            linecounter = linecounter +1
		            if sys.argv[1] == '1' or sys.argv[1] == '2':#only instruction
		               if entry[6] != '':
		                  instance.append(str(getIndex(entry[6])))
		            if len(entry) >= 8:
		               myarg = getArg(entry[6],entry[7])
		               if myarg != None:
		                  if sys.argv[1] == '0' or sys.argv[1] == '2':#only arg
		                     if myarg != '':
		                        instance.append(str(getIndex(myarg)))
		#line_instance = " ".join(list(set(instance[0:int(sys.argv[3])-2])))#num of commands
                line_instance = " ".join(instance)#num of commands
		appendToFile("./outfile"+str(i)+"_"+str(k),line_instance+ " -1 ")
		appendToFile("./outfile"+str(i)+"_"+str(k),"-2\n")

for i in dataset:
    for k in range(0,numinstance):
	    #os.system("java -Xmx2g -jar spmf.jar run TKS "+ "./outfile"+str(i)+"_"+str(k)+" output"+str(i)+ "_" +str(k)+ " 1000")               
	    os.system("java -Xmx2g -jar spmf.jar run CM-SPAM "+ "./outfile"+str(i)+"_"+str(k)+" output"+str(i)+"_"+str(k)+ " " +sys.argv[2]+ "% ")



