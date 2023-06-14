import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import matplotlib.pyplot as plt

# TiTanGPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights2") is False:
        os.makedirs("./weights2")
    if os.path.exists("./best_weights2") is False:
        os.makedirs("./best_weights2")

  #  if os.path.exists("./best_weights") is False:
       # os.makedirs("./best_weights")

    train_images_path, train_images_label, val_images_path, val_images_label, ceshi_images_path, ceshi_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = densenet121(num_classes=args.num_classes).to(device)
    if os.path.exists(args.weights):
        load_state_dict(model, args.weights)

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    T_losses = []
    train_accurate = []
    val_accurate = []
    save_path = './DenseNet121_bs=100八分类.pth'
    print(model.buffers)
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        # train_acc
        train_acc = evaluate(model=model,
                       data_loader=train_loader,
                       device=device)
        print("[epoch {}] train_accuracy: {}".format(epoch, round(train_acc, 3)))
        print('train_acc1 = {}, train_acc2 = {}'.format(train_acc, round(train_acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        scheduler.step()

        train_accurate.append(train_acc)

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        print('acc1 = {}, acc2 = {}'.format(acc, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights2/model-{}.pth".format(epoch))

        if round(acc, 3) > best_acc:
            best_acc = round(acc, 3)
            torch.save(model.state_dict(), "./best_weights2/model-{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)

        T_losses.append(mean_loss)
        val_accurate.append(acc)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc))  # Best_ACC: 0.9960

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(T_losses, label="Densenet121_train_loss")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('./loss_curve_Densenet121八分类.png')
    # 显示损失图
    plt.show()

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(train_accurate, label="Densenet121_train_accurate")
    plt.plot(val_accurate, label="Densenet121_val_accurate")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("accurate")
    plt.legend()
    # 保存损失图
    plt.savefig('./train_val_accurate_Densenet121八分类.png')
    # 显示损失图
    plt.show()

    print('Finished Training')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="E:/python_project/Test8_densenet/data_set/breast_data/breast_photos")

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--weights', type=str, default='densenet121-a639ec97.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
