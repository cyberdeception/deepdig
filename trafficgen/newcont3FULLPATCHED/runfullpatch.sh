#!/bin/bash
mkdir /home/fullpatch

for i in  1 2 3 4 5 6 7 8 9 10 11 12 20
do
./attacker$i.sh 0 200 

#frontend attack 1

mkdir /home/fullpatch/netattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.cap /home/fullpatch/netattacker$i
mkdir /home/fullpatch/sysattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.scap /home/fullpatch/sysattacker$i


ssh -i ./server.pem softseclab@10.176.147.83 rm stream*
sleep 2
done





