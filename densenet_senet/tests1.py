import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
# from utils import ImbalancedDatasetSampler, EarlyStopping, plot_loss
from model import densenet201, densenet161
import torchvision.models.resnet
# from  numpy import np
# from utils import EarlyStopping, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# from __future__ import print_function, division
# import numpy as np
import torch
import os
from torch.utils.data import DataLoader

from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
# from utils import EarlyStopping, plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from models1.se_densenet_full import densenet121, load_state_dict
from model import densenet121
# from confusion_matrix1 import ConfusionMatrix

################################################################################
# 验证模型
################################################################################
def test():
    # 设置超参数
    VAL_PATH = 'Data/test_set.txt'  # 验证集文件地址
    BATCH_SIZE = 32  # Batch size
    VOTE = 'soft'  # 投票策略
    N_CLASS = 8  # 类别数量

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "ceshi": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(
        os.path.join(os.getcwd(), "E:\python_project\Test8_densenet"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "breast_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "ceshi"),
                                            transform=data_transform["ceshi"])

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=100, shuffle=False,
                                                  num_workers=0)

    # 定义模型
    model_list = []
    #assert os.path.isdir('Model'), 'Error: no Model directory found!'
    model = densenet121(num_classes=8)  # Resnet152模型, 输入224x224
    model.load_state_dict(torch.load('./DenseNet121_bs=100八分类.pth'))  # 加载模型权重
    model_list.append(model.double().to(device))
    # model = inception(pretrained=False) # InceptionV3, 输入299x299
    # model.load_state_dict(torch.load('Model/inception_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))
    # model = efficient(pretrained=False) # EfficientNet, 输入224x224
    # model.load_state_dict(torch.load('Model/efficient_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))
    #model_weight_path = "SEdensenet121_full恶性四分类.pth"

    #assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #model.load_state_dict(torch.load(model_weight_path, map_location=device))
    #model_list.append(model.double().to(device))
   ############################
    # 验证
    ############################
    # 测试模式
    model.eval()
    with torch.no_grad():
        # 真实标签 N
        total_y = None
        # 预测标签 N
        total_a = None
        # 预测概率 N x M
        total_b = None
        for data in validate_loader:
            # 加载数据
            x, y = data
            x, y = x.double(), y.long()
            x, y = x.to(device), y.to(device)
            y = y.cpu().detach().numpy()
            y = y.astype(int)
            if total_y is None:
                total_y = y
            else:
                total_y = np.append(total_y, y)

            # 前向传播
            if VOTE == 'soft':
                # 软投票策略
                result = None
                for model in model_list:
                    output = model(x)
                    output = nn.Softmax(dim=1)(output)
                    output = output.detach()
                    if result is None:
                        result = output
                    else:
                        result += output
                result = result / len(model_list)
                # 预测标签
                _, a = torch.max(result.detach(), 1)
                a = a.cpu().detach().numpy()
                a = a.astype(int)
                if total_a is None:
                    total_a = a
                else:
                    total_a = np.append(total_a, a)
                # 预测概率
                b = result.cpu().detach().numpy()
                if total_b is None:
                    total_b = b
                else:
                    total_b = np.concatenate((total_b, b), axis=0)

            else:
                # 硬投票策略
                result = None
                for model in model_list:
                    output = model(x)
                    output = nn.Softmax(dim=1)(output)
                    _, a = torch.max(output.detach(), 1)
                    a = a.cpu().detach().numpy()
                    a = a.astype(int)
                    if result is None:
                        result = a[np.newaxis, :]
                    else:
                        result = np.concatenate((result, a[np.newaxis, :]), axis=0)
                # 根据少数服从多数原则确定每个样本所属类别
                val = []
                for i in range(result.shape[1]):
                    val.append(np.argmax(np.bincount(result[:, i])))
                val = np.asarray(val)
                if total_a is None:
                    total_a = val
                else:
                    total_a = np.append(total_a, val)

        # 计算acc
        acc = accuracy_score(total_y, total_a)
        # 计算f1
        f1 = f1_score(total_y, total_a, average='macro')

    # 打印acc、f1
    print('ACC: %.4f\t F1: %.4f\n'
           % (acc, f1))  # 0.9761




if __name__ == '__main__':
    test()