#!/bin/bash
# Kicks off a run of the applicationDriver. Expects the config file to be used as a parameter.
# Also kills any other instance of java/firefox/Xvfb running

for i in {0..200}
do
# cleanup previous runs
killall firefox Xvfb

# rm the previous nohup.out
rm appdri.out xvfb.out
rm -rf /tmp/*

# start Xvfb
nohup Xvfb :10 -ac > xvfb.out &
export DISPLAY=:10

ssh -i ./debo.pem redherring@104.154.117.255 sudo sysdig -w stream-$i.scap -z -s 4096  container.name=ubuntu&

ssh -i ./debo.pem redherring@104.154.117.255 sudo tcpdump -i eth0 -s0 -w stream-$i.cap port not 22 and port not 3490 and port not 3492 and port not 3790 and port not 80 >pdump.out &

sleep 5
# start the driver asynchronously

nohup python $1 > appdri.out 
echo "here"
sleep 2

ssh -i ./debo.pem reherring@104.154.117.255 sudo killall -s SIGINT sysdig 
ssh -i ./debo.pem redherring@104.154.117.255 sudo killall -s SIGINT tcpdump

done
