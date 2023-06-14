import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from model import densenet201

import torchvision.models.resnet
import time
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    LR = 1e-4  # 学习率
    EPOCH = 100 # 训练轮数
    BATCH_SIZE = 32  # Batch size
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False  # 是否断点训练
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
        os.path.join(os.getcwd(), "D:/PycharmProject/nfnets-pytorch-main"))  # get data root path
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

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = densenet201(num_classes=8)
    model_weight_path = "densenet201-c1103571.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model_weight = torch.load(model_weight_path, map_location=device)
    #  ckptFile = "googleNet1.pth"
    #  ckpt = torch.load(ckptFile, map_location=device)
    net_dict = net.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
    model_weight = {k: v for k, v in model_weight.items() if (k in net_dict and 'classifier' not in k)}
    # 更新权重
    net_dict.update(model_weight)
    net.load_state_dict(net_dict)
    print(net.buffers)

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
   # params = [p for p in net.parameters() if p.requires_grad]
  #  optimizer = optim.Adam(params, lr=0.0001)
    #optimizer = optim.Adam(params, lr=LR)
    #optimizer = optim.Adam(net.parameters(), lr=0.0003)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    epochs = 100
    best_acc = 0.0
    save_path = './DenseNet201V.pth'
    train_steps = len(train_loader)
    # 记录训练过程中训练集、验证集损失
    T_losses = []
    V_losses = []


    # 训练模型
    for epoch in range(start_epoch, EPOCH):
        start_time = time.time()
        print("开始训练")
        ############################
        # 训练
        ############################
        # 训练模式
        net.train()
        # 迭代次数
        cnt = 0
        # 损失
        sum_loss = 0.0
        train_bar = tqdm(train_loader)
        for data in train_loader:
            cnt += 1

            # 加载数据
            x, y = data
            x, y = x.double(), y.long()
            x, y = x.to(device), y.to(device)

            # 梯度置零
            optimizer.zero_grad()

            # 前向传播
            x = x.type(torch.cuda.FloatTensor)
            output = net(x)
            # Resnet152、EfficientNet
           # loss = criterion(output, y)
            loss = loss_function(output, y)
            # # InceptionV3
            # output, aux = model(x)
            # loss = criterion(output, y) + 0.4 * criterion(aux, y)
            # 后向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            sum_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        t_loss = sum_loss / cnt
        # 保存损失
        T_losses.append(t_loss)
        # 打印日志
        print('[%d/%d]\tLoss_T: %.4f'
              % (epoch+1, EPOCH, t_loss), end='')

        ############################
        # 验证
        ############################
        print("开始验证")
        # 训练模式
        net.eval()
        # 迭代次数
        cnt = 0
        # 损失
        sum_loss = 0.0
        # 真实标签 N
        total_y = None
        # 预测标签 N
        total_a = None
        # 预测概率 N x M
        total_b = None
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for data in validate_loader:
                cnt += 1

                # 加载数据
                x, y = data
                x, y = x.double(), y.long()
                x, y = x.to(device), y.to(device)

                # 前向传播
                x = x.type(torch.cuda.FloatTensor)
                output = net(x)

                # loss
              #  loss = criterion(output, y)
                loss = loss_function(output, y)
                sum_loss += loss.item()

                # 预测和真实标签
                output = nn.Softmax(dim=1)(output)
                _, a = torch.max(output.detach(), 1)
                y = y.cpu().detach().numpy()
                a = a.cpu().detach().numpy()
                y = y.astype(int)
                a = a.astype(int)
                if total_y is None:
                    total_y = y
                else:
                    total_y = np.append(total_y, y)
                if total_a is None:
                    total_a = a
                else:
                    total_a = np.append(total_a, a)
                # 预测概率
                b = output.cpu().detach().numpy()
                if total_b is None:
                    total_b = b
                else:
                    total_b = np.concatenate((total_b, b), axis=0)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        v_loss = sum_loss / cnt
        # 保存损失
        V_losses.append(v_loss)

        # 计算acc
        v_acc = accuracy_score(total_y, total_a)
        if best_acc < v_acc:
            best_acc = v_acc
            torch.save(net.state_dict(), save_path)

        # 打印日志
        print('\tLoss_V: %.4f\tAcc_V: %.4f\t\n[====]Time: %.4f[minute]'
              % (v_loss, v_acc,  (time.time() - start_time) / 60))

    print('Finished Training')


if __name__ == '__main__':
    main()