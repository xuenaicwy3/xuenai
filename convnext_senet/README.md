# Skin_classification
## Background
使用改进的ConvNeXt模型对皮肤镜图像进行自动诊断

## 环境依赖
* python3.7
* pytorch 1.11.0
* torchvision 0.12.0

## Usage
1. 在`train.py`中将`--data-path`设置成数据集的路径，将`--weights`参数设成下载好的预训练权重路径，将`--num_classes`设置成相应的类别数
2. 在`create_confusion_matrix_convnext.py`中将 data_root 设置成测试集的路径，并将`model_weight_path`设置成训练好的模型权重路径，得到相应的混淆矩阵
3. 在`predict.py`中将`model_weight_path`设置成模型权重的路径，将`img_path`设置成需要预测的图片路径，就能得到图片预测结果

## 参考文献
1. Liu Z ,  Mao H ,  Wu C Y , et al. A ConvNet for the 2020s[J]. arXiv e-prints, 2022.
2. Yang L, Zhang R Y, Li L, et al. Simam: A simple, parameter-free attention module for convolutional neural networks[C]//International conference on machine learning. PMLR, 2021: 11863-11874.
3. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[J]. arXiv preprint arXiv:1709.01507, 2017, 7.

## 数据集来源
https://challenge.isic-archive.com/data/

## 目录结构描述
```
ConvNeXt
    │  create_confusion_matrix_convnext.py
    │  my_dataset.py
    │  predict.py
    │  README.md
    │  select_incorrect_samples.py
    │  self_defined_trunc_normal_.py
    │  train.py
    │  utils.py
    │  
    ├─attention_module
    │  │  NormalizationBased_Attention_Module.py
    │  │  SEAttention.py
    │  │  SE_Net.py
    │  └─ simam_module.py
    │       
    ├─data
    │ 
    ├─model
    │  │  model.py
    │  │  se_model.py
    │  │  simam_model0.py
    │  │  simam_model1.py
    │  │  simam_model2.py
    │  └─ simam_se_model.py
    │     
    └─weights
```
