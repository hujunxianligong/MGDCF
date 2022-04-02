# MGDCF
Source code and dataset of the paper "Source code and dataset of the paper "MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering"

## InfoBPR Loss

We propose a simple yet powerful InfoBPR Loss for ranking. We also build an out-of-the-box library for InfoBPR:
+ [https://github.com/CrawlScript/InfoBPR](https://github.com/CrawlScript/InfoBPR)

The InfoBPR support both TensorFlow and PyTorch, and it can be installed with pip.


## Requirements

+ Linux
+ Python 3.7
+ tensorflow == 2.4.1
+ tf_geometric == 0.0.81
+ grecx == 0.0.3
+ tqdm=4.51.0


## Run MGDCF

You can run MGDCF with the following command:
```shell
cd scripts/gnn_speed/${DATASET}
sh $SCRIPT_NAME
```
For example, if you want to run Hetero-MGDCF on yelp, the command is:
```shell
cd scripts/gnn_speed/yelp
sh run_gdcf_HeteroMGDCF_yelp.sh
```
Note that the parameter settings are in the shell scripts, and you should only modify the "gpu_ids" argument.
