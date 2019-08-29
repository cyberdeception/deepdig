#!/bin/bash
theserver=18.222.164.205


for i in $(eval echo {$1..$2})
do
ssh -i ./server.pem softseclab@$theserver  sudo sysdig_cap -w stream-$i.scap -z -s 4096 container.name=target and proc.name!=criu and proc.name!=init and proc.name!=systemd-udevd and proc.name!=upstart-udev-br and proc.name!=upstart-socket- and proc.name!=upstart-file-br and proc.name!=sh and proc.name!=iptables and proc.name!=cat and proc.name!=tcpdump &


ssh -i ./server.pem softseclab@$theserver sudo nohup tcpdump_cap -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 2 
python exploit-CVE-2011-3368.py -r $theserver -p 10001 -g / -d 10.0.3.2 -e 8080

sleep 2


ssh -i ./server.pem softseclab@$theserver sudo killall -s SIGINT sysdig_cap 
ssh -i ./server.pem softseclab@$theserver sudo killall -s SIGINT tcpdump_cap


sleep 2
done
