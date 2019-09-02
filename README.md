# DeepDig
A framework for deception-enhanced IDS trainging and evaluation.

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

