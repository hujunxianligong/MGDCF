<p align="center">
<img src="MGDCF_LOGO.png" width="400"/>
</p>

# MGDCF
# Torch-MGDCF
Source code (TensorFlow) and dataset of the paper "[MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)", which is accepted by IEEE Transactions on Knowledge and Data Engineering (TKDE).


This repository contains the TensorFlow implementation of our paper. The official **PyTorch** implementation of 'MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering' is available in a separate repository, which can be accessed here: [https://github.com/CrawlScript/Torch-MGDCF](https://github.com/CrawlScript/Torch-MGDCF).



## Implementations and Paper Links

+ PyTorch Implementation: [Torch-MGDCF](https://github.com/CrawlScript/Torch-MGDCF)
+ TensorFlow Implementation: [TensorFlow-MGDCF](https://github.com/hujunxianligong/MGDCF)
+ Paper Access:
    - **IEEE Xplore**: [https://ieeexplore.ieee.org/document/10384729](https://ieeexplore.ieee.org/document/10384729)
    - **ArXiv**: [https://arxiv.org/abs/2204.02338](https://arxiv.org/abs/2204.02338)





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

```
@ARTICLE{10384729,
  author={Jun Hu and Bryan Hooi and Shengsheng Qian and Quan Fang and Changsheng Xu},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TKDE.2023.3348537}
}
```

