#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
./attacker$i.sh 0 400

#frontend attack 1
mkdir /home/unpatchedhoneydata/netattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.cap /home/unpatchedhoneydata/netattacker$i
mkdir /home/unpatchedhoneydata/sysattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.scap /home/unpatchedhoneydata/sysattacker$i


sleep 2
ssh -i ./server.pem softseclab@10.176.147.83 rm stream*
ssh -i ./server.pem softseclab@10.176.147.83 sudo rm -f /var/lib/libhp/.mon/timestamp.txt
sleep 2

done





#implement mv for /home/honeydata into folder name /home/honeydata$date
#mkdir /home/honeydata



