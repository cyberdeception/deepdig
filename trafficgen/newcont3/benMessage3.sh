#!/bin/bash


for i in {0..200}
do
sudo sysdig -w stream-$i.scap -z -s 4096 container.name=target  &

sudo nohup tcpdump -i lo -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &
sleep 5
curl -k https://localhost/cgi-bin/ss2


sleep 10  

sudo pkill -P $$



sleep 4
done



