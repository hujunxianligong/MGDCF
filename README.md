<p align="center">
<img src="MGDCF_LOGO.png" width="400"/>
</p>

# MGDCF
Source code and dataset of the paper "Source code and dataset of the paper [MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)


This repository contains the TensorFlow implementation of our paper. The official **PyTorch** implementation of 'MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering' is available in a separate repository, which can be accessed here: [https://github.com/CrawlScript/Torch-MGDCF](https://github.com/CrawlScript/Torch-MGDCF).



## Homepage and Paper

+ Homepage (TensorFlow-MGDCF): [https://github.com/hujunxianligong/MGDCF](https://github.com/hujunxianligong/MGDCF)
+ Homepage (Torch-MGDCF): [https://github.com/CrawlScript/Torch-MGDCF](https://github.com/CrawlScript/Torch-MGDCF) 
+ Homepage (InfoBPR Loss): [https://github.com/CrawlScript/InfoBPR](https://github.com/CrawlScript/InfoBPR)
+ Paper: [MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338) 

## InfoBPR Loss

We propose a simple yet powerful InfoBPR Loss for ranking. We also build an out-of-the-box library for InfoBPR:
+ [https://github.com/CrawlScript/InfoBPR](https://github.com/CrawlScript/InfoBPR)

The InfoBPR support both TensorFlow and PyTorch, and it can be installed with pip.


## Requirements

+ Linux
+ Python 3.7
+ tensorflow == 2.7.0
+ tf_geometric == 0.1.5
+ tf_sparse == 0.0.17
+ grecx >= 0.0.6
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



## Cite

If you use MGDCF in a scientific publication, we would appreciate citations to the following paper:

```
@misc{https://doi.org/10.48550/arxiv.2204.02338,
  doi = {10.48550/ARXIV.2204.02338},
  url = {https://arxiv.org/abs/2204.02338},
  author = {Jun Hu and Bryan Hooi and Shengsheng Qian and Quan Fang and Changsheng Xu},
  keywords = {Social and Information Networks (cs.SI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

