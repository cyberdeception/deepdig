#!/bin/bash


path=/home/bigdata/Downloads/benign
./general.sh autoPost.py
mkdir $path/netbenign4
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign4
mkdir $path/sysbenign4
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign4
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2


./general.sh registerCoupon.py
mkdir $path/netbenign5
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign5
mkdir $path/sysbenign5
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign5
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2

./general.sh registerProducts.py
mkdir $path/netbenign6
mkdir $path/sysbenign6
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign6
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign6
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2

./general.sh orderproduct.py
mkdir $path/netbenign7
mkdir $path/sysbenign7
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign7
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign7
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*

./general.sh CreateSocialPostAction.py
mkdir $path/netbenign8
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign8
mkdir $path/sysbenign8
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign8
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2

./general.sh registerUsers.py
mkdir $path/netbenign9
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign9
mkdir $path/sysbenign9
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign9
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2



./general.sh browseProducts.py 
mkdir $path/netbenign10
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign10
mkdir $path/sysbenign10
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign10
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2

./general.sh BrowseSocialNetWork.py
mkdir $path/netbenign11
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign11
mkdir $path/sysbenign11
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign11
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2

./benmessage6.sh 0 5
mkdir $path/netbenign12
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.cap $path/netbenign12
mkdir $path/sysbenign12
scp -i ./debo.pem ubuntu@54.218.47.176:stream*.scap $path/sysbenign12
sleep 2
ssh -i ./debo.pem ubuntu@54.218.47.176 rm stream*
sleep 2


for i in 1 2
do

./benMessage$i.sh 0 5

#frontend benign 1

mkdir  $path/netbenign$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.cap  $path/netbenign$i
mkdir  $path/sysbenign$i
scp -i ./server.pem softseclab@10.176.147.83:stream*.scap  $path/sysbenign$i
sleep 2
ssh -i ./server.pem softseclab@10.176.147.83 rm stream*

sleep 2
done



