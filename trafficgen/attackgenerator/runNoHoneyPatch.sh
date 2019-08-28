#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10 11 12 20 21 22
do
./attacker$i.sh 150 300 

#frontend attack 1
mkdir /home/unpatched_honeydata/netattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.cap /home/unpatched_honeydata/netattacker$i
mkdir /home/unpatched_honeydata/sysattacker$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.scap /home/unpatched_honeydata/sysattacker$i


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





