# DPPA
Dual-Path Neural Network Extracts Tumor Microenvironment Information from Whole Slide Images to Predict Molecular Typing and Prognosis of Glioma



## Preprocess

Import the experimental virtual environment in conda

```
conda create --name dppa python=3.9
```

enter the environment:

```
conda activate dppa
```

Use the following command to install all the libraries and their dependencies

```
conda install --file requirements.txt
```



## Process raw whole slide images

TADMIL prerprocess

```
cd pre_process
python 1-create_patches_fp.py
python 2-my_feature_extractor.py
python 3-cluster.py
python 4-creat_rs_idx.py
```



SRIQ preprocess

```
cd SRIQ
python 1-slide_cluster.py
python 2-create_patch_class_index.py
python 3-my_map_branch.py
```



## Model train

```
cd model_train
python train.py --task 'IDH'
```



## Test

Pre-trained weights and publicly available data can be download at [Link](https://pan.baidu.com/s/17FnGF1EYL2nTmj12sa3LGw?pwd=7r7j), if not available, please contact the corresponding author.

```
cd model_train
python predict.py --task 'IDH'
```





## predict

The molecular typing prediction of IDH can be performed after the previous preprocessing of the sections that need to be predicted

```
cd model_train
python test.py --test_txt './test.txt'
```

