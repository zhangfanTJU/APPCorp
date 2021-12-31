# APPCorp

This repository contains the corpus and code for **APPCorp**.

[**APPCorp: A Corpus for Android Privacy Policy Document Structure Analysis**](https://github.com/zhangfanTJU/APPCorp)

Shuang Liu, Fan Zhang, Baiyang Zhao, Renjie Guo, Tao Chen and Meishan Zhang

## Introduction

**APPCorp** is a manually labelled corpus containing 231 privacy policies (of more than 566K words and 7, 748 annotated paragraphs). We benchmark the corpus with 3 different document classification models, i.e., Support Vector Machine (SVM), Hierarchical Attention Network (HAN) and Hierarchical Graph Attention Network (HGAT), with two different word representations, i.e., GloVe and BERT.

## Reqirements
```
pip install -r requirements.txt
```

## Quick Start
```python
# SVM
python train-svm.py --fold 9

# GloVe
## HAN
python train.py --config_file glove.cfg --emb glove --gpu 0 --fold 9

## HGAT
python train.py --config_file glove.cfg --emb glove --use_graph --gpu 0 --fold 9

# BERT
## HAN
python train.py --config_file bert.cfg --emb bert --gpu 0 --fold 9

## HGAT
python train.py --config_file bert.cfg --emb bert --use_graph --gpu 0 --fold 9
```

## Citation

```bibtex
@article{liu2021appcorp,
  title={APPCorp: A Corpus for Android Privacy Policy Document Structure Analysis}, 
  author={Shuang Liu, Fan Zhang, Baiyang Zhao, Renjie Guo, Tao Chen and Meishan Zhang}
}
```


