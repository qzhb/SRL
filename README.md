# Self-Regulated Learning for Egocentric Video Activity Anticipation

# Introduction 

This is a Pytorch implementation of the model described in our paper:
> Z. Qi, S. Wang, C. Su, L. Su, Q. Huang, and Q. Tian. Self-Regulated Learning for Egocentric Video Activity Anticipation. TPAMI 2021.

# Dependencies
- Pytorch >= 1.0.1
- Cuda 9.0.176
- Cudnn 7.4.2
- Python 3.6.8

# Data

## EPIC-Kitchens dataset
For the raw data of the EPIC-Kitchens dataset, please refer to https://github.com/epic-kitchens/download-scripts to download.

For the three modality features (rgb, flow, obj), please refer to https://github.com/fpv-iplab/rulstm to download. After downloading, put them in the folder './data'.

## EGTEA Gaze+ dataset
Please refer to https://github.com/fpv-iplab/rulstm to download the features. After downloading, put them in the folder './data'.

## 50 Salads dataset


## Breakfast dataset

Please download the extraced I3D features from [Baidu](https://pan.baidu.com/s/1BIbMFlI_gZQrXu1w-EZrhg] passward: wcjv' or [Google Drive](). After downloading, put them in the folder './data'.

# Train

For rgb feature, 
python main.py --gpu_ids 0 --batch_size 128 --wd 1e-5 --lr 0.1 --reinforce_verb_weight 0.01 --reinforce_noun_weight 0.01 --revision_weight 0.8  --mode train --modality rgb --hidden 1024 --feat_in 1024

Silimar commonds can be used for flow or obj features.

# Validation

Please download the pre-trained model weigths from 'https://pan.baidu.com/s/1BIbMFlI_gZQrXu1w-EZrhg passward: wcjv', and put them in the folder './results/EPIC/base_srl/pre_trained/'.

For rgb feature, 
python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality rgb --hidden 1024 --feat_in 1024 --resume_timestamp pre_trained

For flow feature, 
python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality flow --hidden 1024 --feat_in 1024 --resume_timestamp pre_trained

For obj feature, 
python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality obj --hidden 352 --feat_in 352 --resume_timestamp pre_trained

For three modality features, 
python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality fusion --resume_timestamp pre_trained

# Citation
Please cite our paper if you use this code in your own work:
'''

'''
