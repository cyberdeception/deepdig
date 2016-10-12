import math
import os
dataset = [1,2,3,4,5,6,7,8,9,10,11,12,20]
os.system("rm ./entropy")
for i in dataset: 
    os.system("rm ./entropy"+str(i))
os.system("rm dataset.csv")
def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")
dictCommand = []
for i in range(0,22):
    dictCommand.append({})
for i in dataset: 
        filename = "./output"+str(i)
	with open(filename, "r") as f:
	     for line in f:
                 commandkeys = []
                 line = line.rstrip()
		 entry = line.split("#SUP:")
                 commandString = entry[0]
                 sup = entry[1]
                 commandlist = commandString.split(" ")
                 for cmd in commandlist:
                     if cmd != "-1" and cmd !='' and cmd !=" ":
                        commandkeys.append(cmd)
                 commandkey = "_".join(commandkeys)
                 dictCommand[i][commandkey]  = int(sup) 
listkeys = []
for i in dataset: 
    for key in dictCommand[i]:
        listkeys.append(key)
appendToFile("dataset.csv",",".join(listkeys)+",class")
full = len(listkeys)
print full
for i in dataset:
	 for k in range(0,7):
                print k
                mylist = []
                for key in listkeys:
                    mylist.append(str(0))
                span = full/6
                print span
                start = k * span
                end   = (k+1)* span
                print start, end
                if end > full:
                   end = full
		for j in range(start,end):
		    if listkeys[j] in dictCommand[i]:
		       mylist[j] = str(dictCommand[i][listkeys[j]])
		    else:
		       mylist[j] = str(0)
		appendToFile("dataset.csv",",".join(mylist)+ ","+ str(i))

#print dictCommand    

def CalcEntropy(thekeyvaluesFromOtherClass):
    sumkeyvalues=0 
    entropy = 0
    sumkeyvalues = sum(thekeyvaluesFromOtherClass)
    for value in thekeyvaluesFromOtherClass:
        entropy = entropy + (-(float(value)/float(sumkeyvalues)) * math.log(float(value)/float(sumkeyvalues),2))
    return entropy
  

EntropyDictCommand = []
for i in range(0,22):
    EntropyDictCommand.append({})
for i in dataset: 
    for key in dictCommand[i]:
        keyvalue = dictCommand[i][key]
        keyvaluesFromOtherClass = []
        keyvaluesFromOtherClass.append(keyvalue)
        for j in dataset:
            if j != i:
                 if key in  dictCommand[j]:
                     keyvaluesFromOtherClass.append(dictCommand[j][key])
        EntropyDictCommand[i][key] = CalcEntropy(keyvaluesFromOtherClass)
            


#print EntropyDictCommand
from collections import OrderedDict
for i in dataset: 
    tempDict = OrderedDict(sorted(EntropyDictCommand[i].items(), key=lambda t: t[1]))
    for key in tempDict:
        keys = key.split("_")
        appendToFile("entropy"+str(i)," ".join(keys)+ " #SUP: "+ str(tempDict[key]))
        	 
        




