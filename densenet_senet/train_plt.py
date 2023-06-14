import os
import json


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tqdm.contrib import itertools

from model import densenet201
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, confusion_matrix  # 生成混淆矩阵函数
import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import torch.nn as nn
import torch
from h5py.h5t import cfg

# TiTanGPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "E:\python_project\Test8_densenet"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "breast_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=3)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 100
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
    
    net = densenet201()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    #model_weight_path = "./densenet201-c1103571.pth"
    #assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #net.load_state_dict(torch.load(model_weight_path, map_location=device), False)
    #model = models.googlenet(aux_logits=False, pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckptFile = "./densenet201-c1103571.pth"
    ckpt = torch.load(ckptFile, map_location=device)
    net_dict = net.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
    ckpt = {k: v for k, v in ckpt.items() if (k in net_dict and 'fc' not in k)}
    # 更新权重
    net_dict.update(ckpt)
    net.load_state_dict(net_dict)
    #model.load_state_dict(ckpt, False)
    #only_train_fc = True
    #    if only_train_fc:
    # 前面的backbone保持不变
    # for param in model.parameters():
    # param.requires_grad = False

    # print(model.buffers)
    classifier_in_features = net.classifier.in_features
    net.classifier = torch.nn.Linear(classifier_in_features, 4, bias=True)
    print('###############################################################')
    print(net.buffers)
    net = net.to(device)
    net.to('cuda')
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    #in_channel = net.fc.in_features
    #net.fc = nn.Linear(in_channel, 2)
    #print(net.buffers)
    #net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)

    epochs = 5
    best_acc = 0.0
    save_path = './densenet201恶性四分类.pth'
    train_steps = len(train_loader)

    # 记录训练过程中训练集、验证集损失
    T_losses = []
    V_losses = []
    val_accurate1 = []
    train_accurate1 = []

    for epoch in range(epochs):
        # train
        net.train()
        train_acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            # 准确率
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            #####
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num
        train_accurate1.append(train_acc / train_num)

        # 保存损失
        T_losses.append(running_loss / train_steps)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate))

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
        val_accurate1.append(acc / val_num)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc))  # Best_ACC: 0.9862

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(T_losses, label="train_loss")
 #   plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('./loss_curve_densenet.png')
    # 显示损失图
    plt.show()

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(val_accurate1, label="val_accurate")
    plt.plot(train_accurate1, label="train_accurate")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("accurate")
    plt.legend()
    # 保存损失图
    plt.savefig('./train_val_accurate_densenet.png')
    # 显示损失图
    plt.show()

    print('Finished Training')





if __name__ == '__main__':
    main()
