#!/bin/bash


for i in $(eval echo {$1..$2})
do
ssh -i ./server.pem softseclab@10.176.147.83  sudo sysdig_cap -w stream-$i.scap -z -s 4096 container.name=target and proc.name!=criu and proc.name!=init and proc.name!=systemd-udevd and proc.name!=upstart-udev-br and proc.name!=upstart-socket- and proc.name!=upstart-file-br and proc.name!=sh and proc.name!=iptables and proc.name!=cat &


ssh -i ./server.pem softseclab@10.176.147.83 sudo nohup tcpdump_cap -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 5 
python heartbleed4.py 10.176.147.83
python heartbleed4.py 10.176.147.83

sleep 5

ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT sysdig_cap 
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT tcpdump_cap



sleep 4
done
