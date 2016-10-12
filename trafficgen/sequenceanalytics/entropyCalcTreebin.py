import math
import os
import sys

dataset = [0,1] #0 ben 1 att


numinstance = 10
'''os.system("rm ./entropy")
for i in dataset: 
    os.system("rm ./entropy"+str(i))
'''
os.system("rm dataset.csv")
def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")
dictCommand = []
for k in range(0,numinstance):
    dictCommand.append([])

for k in range(0,numinstance):
        for i in range(0,3):
            dictCommand[k].append({})
for i in dataset: 
        for k in range(0,numinstance):
        	filename = "./output"+str(i)+"_"+str(k)
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
		         dictCommand[k][i][commandkey]  = int(sup) 
listkeys = []
for i in dataset:
    for k in range(0,numinstance): 
	    for key in dictCommand[k][i]:
		listkeys.append(key)

newlistkeys = list(set(listkeys))
appendToFile("dataset.csv",",".join(newlistkeys)+",class")
full = len(newlistkeys)
print full
for i in dataset:
	 for k in range(0,numinstance):
                mylist = []
                for key in newlistkeys:
                    if key in dictCommand[k][i]:
		       mylist.append(str(dictCommand[k][i][key]))
		    else:
		       mylist.append(str(0))
		appendToFile("dataset.csv",",".join(mylist)+ ","+ str(i))


#print dictCommand    
'''
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
        	 
'''        




