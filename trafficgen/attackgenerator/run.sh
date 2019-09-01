#!/bin/bash
theserver=18.222.164.205
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17-22 
do
./attacker$i.sh 0 5

#frontend attack 1
mkdir /home/honeydata/netattacker$i
scp -i ./server.pem softseclab@$theserver:stream*.cap /home/honeydata/netattacker$i
mkdir /home/honeydata/sysattacker$i
scp -i ./server.pem softseclab@$theserver:stream*.scap /home/honeydata/sysattacker$i


#decoy attack 1
mkdir /home/honeydata/netattackerdecoy$i
scp -i ./server.pem softseclab@$theserver:/var/lib/libhp/.mon/*CVE*.pcap /home/honeydata/netattackerdecoy$i
mkdir /home/honeydata/sysattackerdecoy$i
scp -i ./server.pem softseclab@$theserver:/var/lib/libhp/.mon/*CVE*.scap /home/honeydata/sysattackerdecoy$i
sleep 2
ssh -i ./server.pem softseclab@$theserver rm stream*
ssh -i ./server.pem softseclab@$theserver sudo rm -f /var/lib/libhp/.mon/*CVE*
sleep 2

done





#implement mv for /home/honeydata into folder name /home/honeydata$date
#mkdir /home/honeydata



