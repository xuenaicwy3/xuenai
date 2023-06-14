import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchviz import make_dot

from my_dataset import MyDataSet
#from model.model import convnext_tiny as create_model
from model.simam_se_model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
import matplotlib.pyplot as plt

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parmeters:{total_params}")

    save_path = './SEConvNext_T.pth'
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    T_losses = []
    V_losses = []
    train_accurate = []
    val_accurate = []
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        print("[epoch {}] train_accuracy: {}".format(epoch, round(train_acc, 3)))
        print('train_acc1 = {}, train_acc2 = {}'.format(train_acc, round(train_acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        train_accurate.append(train_acc)
        T_losses.append(train_loss)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))
        print('acc1 = {}, acc2 = {}'.format(val_acc, round(val_acc, 3)))
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if round(val_acc, 3) > best_acc:
            best_acc = round(val_acc, 3)
            #torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)

        V_losses.append(val_loss)
        val_accurate.append(val_acc)

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train Loss During Training")
    plt.plot(T_losses, label="train_loss")
    #plt.plot(V_losses, label="val_loss")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("epochs")
    plt.ylabel("train_val_Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('./seCovNext-T.png')
    # 显示损失图
    plt.show()

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val accurate During Training")
    #plt.plot(val_accurate, label="val_accurate")
    plt.plot(train_accurate, label="train_accurate")
    #   plt.plot(V_losses, label="V")
    plt.xlabel("epochs")
    plt.ylabel("accurate")
    plt.legend()
    # 保存损失图
    plt.savefig('./seCovNext-T_.png')
    # 显示损失图
    plt.show()

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # # 数据集所在根目录
    # parser.add_argument('--data-path', type=str, default="D:/数据集/40X")
    #
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # # 是否冻结head以外所有权重
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    #
    # opt = parser.parse_args()
    #
    # main(opt)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/数据集/40X")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)