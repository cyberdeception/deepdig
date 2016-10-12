#!/bin/bash

#for i in 1 2 3 4 5 6 7 8 9 10 11 12
for i in 6
do
./attacker$i.sh 0 150 
ssh -i ./server.pem softseclab@10.176.147.83 python extractData.py


#frontend attack 1
mkdir /home/honeydata/netattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.cap /home/honeydata/netattacker$i
mkdir /home/honeydata/sysattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.scap /home/honeydata/sysattacker$i


#decoy attack 1
#mkdir /home/honeydata/netattackerdecoy$i
#scp -i ./server.pem softseclab@10.176.147.83:/var/lib/libhp/.mon/*CVE*.pcap /home/honeydata/netattackerdecoy$i
#mkdir /home/honeydata/sysattackerdecoy$i
#scp -i ./server.pem softseclab@10.176.147.83:/var/lib/libhp/.mon/*CVE*.scap /home/honeydata/sysattackerdecoy$i
#sleep 2
ssh -i ./server.pem softseclab@10.176.147.83 rm stream*
ssh -i ./server.pem softseclab@10.176.147.83 sudo rm -f /var/lib/libhp/.mon/timestamp.txt
sleep 2

done





#implement mv for /home/honeydata into folder name /home/honeydata$date
#mkdir /home/honeydata



