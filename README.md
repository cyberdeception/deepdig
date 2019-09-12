# DeepDig
A framework for deception-enhanced IDS training and evaluation.

## Network Traffic Generation
The platform can be used to generate _attack_ and _benign_ traffic and evaluation data.

Supported attack types are described below. Additional attacks can be customized in the framework.

|\#     | Attack Type   | Description                  | Software |
|------ |---------------|------------------------------|----------|
| 1     | CVE-2014-0160 | Information leak             | OpenSSL  |
| 2     | CVE-2012-1823 | System remote hijack         | PHP      |
| 3     | CVE-2011-3368 | Port scanning                | Apache   |
| 4–10  | CVE-2014-6271 | System hijack (7 variants)   | Bash     | 
| 11    | CVE-2014-6271 | Remote Password file read    | Bash     | 
| 12    | CVE-2014-6271 | Remote root directory read   | Bash     | 
| 13    | CVE-2014-0224 | Session hijack and info leak | OpenSSL  | 
| 14    | CVE-2010-0740 | DoS via NULL pointer deref   | OpenSSL  | 
| 15    | CVE-2010-1452 | DoS via request lacking path | Apache   |
| 16    | CVE-2016-7054 | DoS via heap buffer overflow | OpenSSL  | 
| 17–22 | CVE-2017-5941 | System hijack (6 variants)   | Node.js  |

### Attack generation
> Note: the IP address of the target servers can be modified in each attack script 

> Note: the IP addresses will have to point to a server with vulnerabilities listed above which has is not provided to execute attack workload:
```
cd trafficgen/attackgenerator
./run.sh
```


### Benign traffic generation
> Pre-requisite: the benign traffic generators need a wordpress deployment. It also requires the Buddypress and Woocommerce plugins for creating application profiles. 

> Please install firefox web browser if it is not available on your machine.

Install selenium for Python: 
```
pip install selenium 
```
Run traffic generator: 
```
cd trafficgen/benignGenerator
./general.sh
```

## ML Module & Experiments 
Change dir to `ml` submodule:
`cd ml` 

Download and decompress the datasets from the link below:
> https://drive.google.com/drive/folders/1Skrzw62SC5X8qAWcHVL9k8BpMTI4TNYG?usp=sharing

Create data directory:
```
mkdir data
```


For the OML experiments:
```
tar -C data -xvzf datafiles\_oml.tar.gz


### Online Metric Learning (OML)
Build container:
```
docker build -t oml -f Dockerfile.oml .




```
Run interactive shell into container:
```
docker run -it --rm -v data:/workspace/datafiles_oml oml bash
```
docker ps
``` 

get the id of the container

```
copy the data to the container by running this command on the host machine.
```
sudo docker cp data/datafiles_oml/ [yourcontainerid]:/workspace

```
To run experiments:
Inside the container prompt
```
```
cd /workspace/code/oml
```
./runincalltest\_single.sh > output
```
cat output | grep Acc
```
cat output | grep FPR
```
cat output | grep TPR
```
cat output | grep F2
```
./runincalltesthuman\_single.sh > outputh
```
cat outputh | grep Acc
```
cat outputh | grep FPR
```
cat outputh | grep TPR
```
cat outputh | grep F2
```


```

For the SVM experiments: 
```
cat data\_svm\_split.tgz\_* | tar -C data -xz
```


### SVM
Build container:
```
docker build -t svm -f Dockerfile.svm .
```
Run interactive shell into container:
```
docker run -it --rm -v data:/workspace/datafiles oml bash
```
To run experiments:
```
cd /workspace/code/svm
./scriptme\_16.sh
python parseResultsFile.py
```
Check `output` folder

