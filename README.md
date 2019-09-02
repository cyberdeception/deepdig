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
To execute attack workload:
```
cd trafficgen/attackgenerator
./run.sh
```
> Note: the IP address of the target servers can be modified in each attack script 

### Benign traffic generation
> Pre-requisite: the benign traffic generators need a wordpress deployment. It also requires the Buddypress and Woocommerce plugins for creating application profiles.
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

For the SVM experiments: 
```
cat data\_svm\_split.tgz\_\* | tar -C data -xz
```

For the OML experiments:
```
tar -C data -xvzf datafiles\_oml.tar.gz
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

### Online Metric Learning (OML)
Build container:
```
docker build -t oml -f Dockerfile.oml .
```
Run interactive shell into container:
```
docker run -it --rm -v data:/workspace/data oml bash
```
To run experiments:
```
cd /workspace/code/oml
./runincalltest\_single.sh
./runincalltesthuman\_single.sh
```

