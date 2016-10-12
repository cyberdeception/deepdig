#!/bin/bash


for i in $(eval echo {$1..$2})
do
python heartbleed4.py 10.176.147.83
python heartbleed4.py 10.176.147.83

sleep 2 

ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT sysdig_cap 
ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT tcpdump_cap
done
