#!/bin/bash
# Kicks off a run of the applicationDriver. Expects the config file to be used as a parameter.
# Also kills any other instance of java/firefox/Xvfb running

# cleanup previous runs
killall firefox Xvfb

# rm the previous nohup.out
rm appdri.out xvfb.out
rm -rf /tmp/*

# start Xvfb
nohup Xvfb :10 -ac > xvfb.out &
export DISPLAY=:10

# start the driver asynchronously

nohup python registerUsers.py > appdri.out &


