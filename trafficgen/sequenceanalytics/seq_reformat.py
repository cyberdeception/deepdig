
import os



os.system("rm ./reformated.txt")

def appendToFile(filename,line):
    with open(filename, "a") as myfile:
         myfile.write(line+"\n")
 

filename = "./index_map"
instmap = {}

with open(filename, "r") as f:
     for line in f:
         line = line.rstrip()
         entry = line.split(" ")
         instmap[int(entry[0])] = entry[1]

filename = "./outtext"
with open(filename, "r") as f:
     for line in f:
         line = line.rstrip()
         entry = line.split(" ")
         outline = []
         for j in range(0,len(entry)):
             if entry[j] == "-1":
                continue
             if entry[j] != "==>" and entry[j] != "#SUP:":
               outline.append(instmap[int(entry[j])])
             if entry[j] == "==>":
                 outline.append(entry[j])
             if entry[j] == "#SUP:":
                for k in range(j,len(entry)):
                    outline.append(entry[k])
                break;
         appendToFile("reformated.txt"," ".join(outline))


        




