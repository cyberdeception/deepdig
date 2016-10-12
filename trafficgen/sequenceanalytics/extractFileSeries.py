
def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")

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
    for i in [4]:
      for j in range(150,200):
        filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
        #filename = "stream-6.scap"
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

for j in range(0,len(inst)):
    print str(j), inst[j]


def getIndex(instname):
    for i in range(0,len(inst)):
        if inst[i] == instname:
           return i
    return 0;
    


for i in [4]:
    for j in range(150,200):
        instance = []
        filename = "./sysattacker"+str(i)+"/stream-"+str(j)+".scap"
        #filename = "stream-6.scap"
        with open(filename, "r") as f:
             for line in f:
                 line = line.rstrip()
                 entry = line.split(" ")
                 if entry[4] != "tcpdump":
                    instance.append((entry[6]))
                    if len(entry) >= 8:
                       myarg = getArg(entry[6],entry[7])
                       if myarg != None:
                          instance.append(myarg)
        line_instance = "\n".join(instance)
        appendToFile("./outfile"+ str(i)+str(j),line_instance)
        
                 
            
