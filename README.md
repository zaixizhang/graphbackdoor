# BackdoorGNN
A PyTorch implementation of "Backdoor Attacks to Graph Neural Networks" (SACMAT'21) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) [[arxiv]](https://arxiv.org/abs/2006.11165)

## Abstract   
In this work, we propose the first backdoor attack to graph neural
networks (GNN). Specifically, we propose a *subgraph based backdoor
attack* to GNN for graph classification. In our backdoor attack, a
GNN classifier predicts an attacker-chosen target label for a testing
graph once a predefined subgraph is injected to the testing graph.
Our empirical results on three real-world graph datasets show
that our backdoor attacks are effective with a small impact on
a GNN’s prediction accuracy for clean testing graphs. Moreover,
we generalize a randomized smoothing based certified defense to
defend against our backdoor attacks. Our empirical results show
that the defense is effective in some cases but ineffective in other
cases, highlighting the needs of new defenses for our backdoor
attacks.

## Requirements

```
matplotlib==3.1.1
numpy==1.17.1
torch==1.2.0
scipy==1.3.1
networkx==2.4
```

## Cite

If you find this repo to be useful, please cite our paper. Thank you.

```
@inproceedings{10.1145/3450569.3463560,
author = {Zhang, Zaixi and Jia, Jinyuan and Wang, Binghui and Gong, Neil Zhenqiang},
title = {Backdoor Attacks to Graph Neural Networks},
year = {2021},
isbn = {9781450383653},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450569.3463560},
doi = {10.1145/3450569.3463560},
booktitle = {Proceedings of the 26th ACM Symposium on Access Control Models and Technologies},
pages = {15–26},
numpages = {12},
keywords = {graph neural networks, backdoor attack},
location = {Virtual Event, Spain},
series = {SACMAT '21}
}
```
