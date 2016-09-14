#!/bin/bash


for i in $(eval echo {$1..$2})
do
ssh -i ./debo.pem ubuntu@104.154.117.255 sudo nohup sysdig -w stream-$i.scap -z -s 4096 container.name=ubuntu &

ssh -i ./debo.pem ubuntu@104.154.117.255 sudo -A nohup tcpdump -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 6
curl -k https://104.154.117.255/wordpress/

sleep 4
ssh -i ./debo.pem ubuntu@104.154.117.255 sudo killall -s SIGINT sysdig 
ssh -i ./debo.pem ubuntu@104.154.117.255 sudo killall -s SIGINT tcpdump

sleep 4
done


