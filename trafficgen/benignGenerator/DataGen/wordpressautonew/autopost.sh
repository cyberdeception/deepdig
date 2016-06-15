#!/bin/bash
# Kicks off a run of the applicationDriver. Expects the config file to be used as a parameter.
# Also kills any other instance of java/firefox/Xvfb running

for i in {1..200}
do
# cleanup previous runs
killall firefox Xvfb

# rm the previous nohup.out
rm appdri.out xvfb.out
rm -rf /tmp/*

# start Xvfb
nohup Xvfb :10 -ac > xvfb.out &
export DISPLAY=:10

ssh -i ./debo.pem ubuntu@54.218.47.176 sudo nohup sysdig -w stream-$i.scap &

ssh -i ./debo.pem ubuntu@54.218.47.176 sudo -A nohup tcpdump -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 5
# start the driver asynchronously
nohup python autoPost.py > appdri.out 

sleep 5

ssh -i ./debo.pem ubuntu@54.218.47.176 sudo nohup killall -9 sysdig 
ssh -i ./debo.pem ubuntu@54.218.47.176 sudo nohup killall -9 tcpdump


done


