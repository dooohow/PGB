# PGB
## Requirements
```
numpy >= 1.20.1
pandas >= 1.2.4
networkx >= 2.5
scikit-learn >= 0.24.1
python-louvain >= 0.15
python >= 3.8
scipy >= 1.10.1
```

## Contents
1. data (folder): All datasets are in this folder.
2. comm (folder): This folder is used for community discovery.
3. utils.py (file): The file includes some functions that are needed for other files.
4. graph results (folder): Line graph based on experimental results.
5. result (folder): This folder is used to store the results.
6. DGG.py (file): The file is used to obtain the results of DGG for experiments.
7. DP-1K.py (file): The file is used to obtain the results of DP-dK for experiments.
8. PrivGraph.py (file): The file is used to obtain the results of PrivGraph for experiments.
9. SKG.py (file): The file is used to obtain the results of DP-SKG for experiments.
10. Tmf.py (file): The file is used to obtain the results of Tmf for experiments.

## Run
###### Example: End to End ######
```
python Tmf.py
```

## Execution Environment
### CPU：AMD EPYC 7313P 16cores 32threads 3.7Ghz
### Memory: 503GB
### OS: Ubuntu 20.04.5 LTS
