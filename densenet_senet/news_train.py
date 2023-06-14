import os
import argparse
import math


import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from my_dataset import MyDataSet
from model import densenet121 as create_model, load_state_dict
from models1.se_densenet_transition import densenet121 as create_model1, load_state_dict
from models1.se_densenet_block import densenet121 as create_model2, load_state_dict
from models1.se_densenet_full import densenet121 as create_model3, load_state_dict
from utils import read_split_data, train_one_epoch, evaluate
import matplotlib.pyplot as plt

# TiTanGPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
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

    model = create_model(num_classes=args.num_classes).to(device)
    model1 = create_model1(num_classes=args.num_classes).to(device)
    model2 = create_model2(num_classes=args.num_classes).to(device)
    model3 = create_model3(num_classes=args.num_classes).to(device)

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

    best_acc = 0.
    best_acc1 = 0.
    best_acc2 = 0.
    best_acc3 = 0.
    T_losses = []
    V_losses = []
    val_accurate1 = []
    val_accurate = []
    val_accurate2 = []
    val_accurate3 = []
    save_path = './Densenet121八分类.pth'
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

        torch.save(model.state_dict(), "./weights3/model-{}.pth".format(epoch))

        if round(acc, 3) > best_acc:
            best_acc = round(acc, 3)
            torch.save(model.state_dict(), "./best_weights3/model-{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)

        T_losses.append(mean_loss)
        val_accurate.append(acc)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc))  # se_densenet_tranition: 0.9990
    ####################################################################################

    print(model1.buffers)

    if os.path.exists(args.weights):
        load_state_dict(model1, args.weights)

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model1.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg1 = [p for p in model1.parameters() if p.requires_grad]
    optimizer1 = optim.SGD(pg1, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler1 = lr_scheduler.LambdaLR(optimizer1, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model1,
                                    optimizer=optimizer1,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        # train_acc
        train_acc = evaluate(model=model1,
                       data_loader=train_loader,
                       device=device)
        print("[epoch {}] train_accuracy: {}".format(epoch, round(train_acc, 3)))

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer1.param_groups[0]["lr"], epoch)

        scheduler1.step()


        # validate
        acc = evaluate(model=model1,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer1.param_groups[0]["lr"], epoch)

        if round(acc, 3) > best_acc1:
            best_acc1 = round(acc, 3)

        val_accurate1.append(acc)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc1))  # se_densenet_tranition: 0.9990
    ####################################################################################
    print(model2.buffers)

    if os.path.exists(args.weights):
        load_state_dict(model2, args.weights)

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model2.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg2 = [p for p in model2.parameters() if p.requires_grad]
    optimizer2 = optim.SGD(pg2, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler2 = lr_scheduler.LambdaLR(optimizer2, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model2,
                                    optimizer=optimizer2,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        # train_acc
        train_acc = evaluate(model=model2,
                       data_loader=train_loader,
                       device=device)
        print("[epoch {}] train_accuracy: {}".format(epoch, round(train_acc, 3)))
        print('train_acc1 = {}, train_acc2 = {}'.format(train_acc, round(train_acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer2.param_groups[0]["lr"], epoch)

        scheduler2.step()


        # validate
        acc = evaluate(model=model2,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        print('acc1 = {}, acc2 = {}'.format(acc, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer2.param_groups[0]["lr"], epoch)


        if round(acc, 3) > best_acc2:
            best_acc2 = round(acc, 3)

        val_accurate2.append(acc)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc2))  # se_densenet_tranition: 0.9990
    #####################################################################################

    print(model3.buffers)
    if os.path.exists(args.weights):
        load_state_dict(model3, args.weights)

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model3.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg3 = [p for p in model3.parameters() if p.requires_grad]
    optimizer3 = optim.SGD(pg3, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler3 = lr_scheduler.LambdaLR(optimizer3, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model3,
                                    optimizer=optimizer3,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        # train_acc
        train_acc = evaluate(model=model3,
                             data_loader=train_loader,
                             device=device)
        print("[epoch {}] train_accuracy: {}".format(epoch, round(train_acc, 3)))
        print('train_acc1 = {}, train_acc2 = {}'.format(train_acc, round(train_acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer3.param_groups[0]["lr"], epoch)

        scheduler3.step()

        # validate
        acc = evaluate(model=model3,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer3.param_groups[0]["lr"], epoch)

        if round(acc, 3) > best_acc3:
            best_acc3 = round(acc, 3)

        val_accurate3.append(acc)

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        best_acc3))  # se_densenet_tranition: 0.9990


    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train Loss During Training")
    plt.plot(T_losses, label="train_loss")
    plt.plot(V_losses, label="val_loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('./CConvNext八分类_loss.png')
    # 显示损失图
    plt.show()

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val accurate During Training")
    plt.plot(val_accurate, label="DenseNet121_val_accurate")
    plt.plot(val_accurate1, label="SEDenseNet121_transition_val_accurate")
    plt.plot(val_accurate2, label="SEDenseNet121_block_val_accurate")
    plt.plot(val_accurate3, label="SEDenseNet121_trbl_val_accurate")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("accurate")
    plt.legend()
    # 保存损失图
    plt.savefig('./val_VAL1_accurate_DenseNet八分类.png')
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
    #parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--wd', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="E:/python_project/Test12_ConvNeXt/data_set/breast_data/breast_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./densenet121-a639ec97.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
