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
#from utils import ImbalancedDatasetSampler, EarlyStopping, plot_loss
from model import densenet201, densenet161
import torchvision.models.resnet
#from  numpy import np
#from utils import EarlyStopping, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#from __future__ import print_function, division
#import numpy as np
import torch
import os
from torch.utils.data import DataLoader

from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
#from utils import EarlyStopping, plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
#from confusion_matrix1 import ConfusionMatrix

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "D:\PycharmProject\Test8_densenet"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=7)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = densenet201(num_classes=8)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    """
    model_weight_path = "densenet201-c1103571.pth"

    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 8)
    """
    print(net.buffers)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)



    epochs = 100
    best_acc = 0.0
    save_path = './densenet201.pth'
    train_steps = len(train_loader)
    acc1 = 0.0
    train_loss = []
    acc_accuracy = []
    for epoch in range(epochs):
        # train
        total, correct = 0, 0                          #total, correct, train_loss = 0, 0 ,0
        net.train()                                    #start = time.time()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):        #for i, (X, y) in enumerate(train_loader)
            images, labels = data                           #X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(images.to(device))                 #output = net(X)
            loss = loss_function(logits, labels.to(device)) #loss = criterion(output, y)
            loss.backward()
            optimizer.step()                                #optimizer.zero_grad()
                                                            #loss.backward()
                                                            #optimizer.step()
            # print statistics
            running_loss += loss.item()                     #train_loss += loss.item()
            total += labels.to(device).size(0)              #total += y.size(0)
            correct += (logits.argmax(dim=1) == labels.to(device)).sum().item() # correct += (logits.argmax(dim=1) == labels.to(device)).sum().item()
            train_acc = 100.0 * correct / total             #train_acc = 100.0 * correct / total

         #   outputs = net(images.to(device))
            # loss = loss_function(outputs, test_labels)
        #    predict_y = torch.max(outputs, dim=1)[1]
         #   acc1 += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_loss.append(running_loss / train_steps)
     #   accurate = acc1 / val_num

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        acc_accuracy.append(acc / val_num)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d] accuracy: %.3f' % (epoch + 1, train_acc / 100))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)


          
        """
        class_indict = {"0": "Mductal_carcinoma", "1": "Mlobular_carcinoma", "2": "Mmucinous_carcinoma", "3": "Mpapillary_carcinoma",
        "4": "adenosis", "5": "fibroadenoma", "6": "phyllodes_tumor", "7": "tubular_adenoma"}
        label = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=8, labels=label)
        #实例化混淆矩阵 这里NUM_CLASSES = 5

        with torch.no_grad():
            net.eval()#验证
            valid_loss= 0.0
            for j, (inputs, labels) in enumerate(validate_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = net(inputs)#分类网络的输出 分类器用的softmax 即使不使用softmax也不影响分类结果
                loss = loss_function(output, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(output.data, 1)#torch.max获取output最大值以及下标 predictions即为预测值
                #confusion_matrix
                confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())

            confusion.plot()
            confusion.summary()
        """


    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(train_loss, label="train_loss")
#    plt.plot(V_losses, label="val_loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('./loss_curve.png')
    # 显示损失图
  #  plt.show()

    # 绘制准确率图
    plt.figure(figsize=(10, 5))
    plt.title("DenseNet")
    plt.plot(acc_accuracy, label="densenet201_accurate")
#    plt.plot(train_accurate, label="train_accurate")
    plt.xlabel("Epoches")
    plt.ylabel("accurate")
    plt.legend()
    # 保存损失图
    plt.savefig('./val_accurate.png')
    # 显示损失图
    plt.show()

    print('Finished Training')

    print('Finished Training')


################################################################################
# 验证模型
################################################################################
def val():
    # 设置超参数
    VAL_PATH = 'Data/test_set.txt'  # 验证集文件地址
    BATCH_SIZE = 32  # Batch size
    VOTE = 'soft'  # 投票策略
    N_CLASS = 8  # 类别数量

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据处理
    normalize = transforms.Normalize(  # 224x224
        mean=[0.7634611, 0.54736304, 0.5729477],
        std=[0.1432169, 0.1519472, 0.16928367]
    )
    # normalize = transforms.Normalize( # 299x299
    #     mean=[0.76359415, 0.5453203, 0.5692775],
    #     std=[0.14003323, 0.15183851, 0.1698507]
    # )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(
        os.path.join(os.getcwd(), "D:\PycharmProjects\pythonProject8\Test8_densenet"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 加载数据
#    val_dataset = SkinDiseaseDataset(VAL_PATH, transforms=val_transform, aug=False)  # 定义valloader
#   val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=16, shuffle=False,
                                                  num_workers=0)

    # 定义模型
    model_list = []
    assert os.path.isdir('Model'), 'Error: no Model directory found!'
    model = resnet34()  # Resnet152模型, 输入224x224
    model.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))  # 加载模型权重
    model_list.append(model.double().to(device))
    # model = inception(pretrained=False) # InceptionV3, 输入299x299
    # model.load_state_dict(torch.load('Model/inception_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))
    # model = efficient(pretrained=False) # EfficientNet, 输入224x224
    # model.load_state_dict(torch.load('Model/efficient_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))

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
        # 计算G-mean
        g_mean = geometric_mean_score(total_y, total_a, average='macro')
        # 计算auc
        auc = 0.
        if VOTE == 'soft':
            auc = roc_auc_score(label_binarize(total_y, np.arange(N_CLASS)), total_b, average='macro')

    # 打印acc、f1
    print('ACC: %.4f\t F1: %.4f\n'
          'G-mean: %.4f\t AUC: %.4f' % (acc, f1, g_mean, auc))

    # 计算混淆矩阵
    cm = confusion_matrix(total_y, total_a)
    # 可视化混淆矩阵
    cm_plot_labels = [["Mductal_carcinoma",
       "Mlobular_carcinoma",
       "Mmucinous_carcinoma",
      "Mpapillary_carcinoma",
     "adenosis",
       "fibroadenoma",
       "phyllodes_tumor",
      "tubular_adenoma"]]
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')





if __name__ == '__main__':
    main()

   # val()