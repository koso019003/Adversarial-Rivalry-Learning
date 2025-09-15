# Adversarial Rivalry Learning for Music Classification


![Model architecture](./figures/ARL_arch.png)
<p align="center"> Proposed genreMERT (with ARL) architecture. </p>

## Introduction

This repo is official PyTorch implementation of **Adversarial Rivalry Learning for Music Classification (submitted to ICASSP 2026)**.

we introduced **adversarial rivalry learning (ARL)**, a novel learning paradigm that overcomes LCA’s limitations by reducing hyperparameter tuning burden and providing a clear learning criterion. Through structured branch rivalry, ARL achieves superior performance across benchmarks with no additional inference cost, offering a practical solution for music classification.


Pleaser refer to our **[paper]()** for further details.


## Getting started 

#### Installation & Clone the repo [Environment on Linux (Ubuntu 22.04 with python >= 3.8)]

```bash
# Clone the repo:
$ git clone ...
$ cd $PWD/Adversarial-Rivalry-Learning

# Install the requirements using `conda` and `pip`: 
$ conda create -y -n genreMERT python=3.8
$ conda activate genreMERT
$ conda install -y pytorch=2.2.1 torchaudio=2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
$ conda install -y lightning=2.2.0.post0 -c conda-forge

$ pip install -r requirements.txt

```

## Download weights 

Download the pre-trained genreMERT (w/ ARL) weights from [here](https://drive.google.com/drive/folders/1RcmSbOKKcR2qmwmjMdX7mt073SFDwt62?usp=sharing).

You need to unzip the contents, and the data directory structure should follow the hierarchy below.

```
${ROOT}  
|-- weights  
|   |-- genreMERT_ARL.ckpt
```

## Evaluation

Run the commands below under the folder src/ to evaluate a pretrained model on GTZAN test set.

You should be able to obtain the output below, which is identical to the result in Table 4 (see our paper):


```bash
$ cd src

# genreMERT (w/ ARL)
$ CUDA_VISIBLE_DEVICES=0 python main.py
Seed set to 0
Loading dataset...
Setting sample rate from 22050 to 24000
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Predicting test_dataset result
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Predicting DataLoader 0: 100%|████████████████████████████████████| 6/6 [00:07<00:00,  4.30it/s]

Number of parameters: 94.40 (M)
Test accuracy of y:
Frame level: 0.90       Song level: 0.93
```

You should also be able to obtain the results under the ROOT/result.

### Results

| Models             | Frame Level Acc. | Song Level Acc. | #Para. (M) |
|--------------------|------------------|-----------------|------------|
| genreMERT (w/ LCA) | 0.89             | 0.92            | 94.40      |
| genreMERT (w/ ARL) | 0.90             | 0.93            | 94.40      |

### Confusion matrix

| genreMERT (w/ LCA) | genreMERT (w/ ARL) |
|--------------------|--------------------|
| <img src="./figures/genreMERT_LCA_CM.png"> | <img src="./figures/genreMERT_ARL_CM.png"> |

## Citation

```bibtex

```

## License
This project is licensed under the terms of the MIT license.

## References
This code is based on [koso019003/Learnable-Counterfactual-Attention](https://github.com/koso019003/Learnable-Counterfactual-Attention) and partly based on [huggingface/transformers](https://github.com/huggingface/transformers).

The pre-trained weights of MERT used in genreMERT (w/ ARL) are from [here](https://huggingface.co/m-a-p/MERT-v1-95M).
