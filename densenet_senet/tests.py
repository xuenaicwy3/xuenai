import os
import math
import argparse
from torchvision import transforms, datasets
import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import densenet121, load_state_dict
#from models1.se_densenet_full import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import matplotlib.pyplot as plt
#from models1.se_densenet_full import densenet121, load_state_dict
# TiTanGPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    data_transform = {
        "ceshi": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(
        os.path.join(os.getcwd(), "E:\python_project\Test8_densenet"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "breast_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "ceshi"),
                                            transform=data_transform["ceshi"])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=100, shuffle=False,
                                                  num_workers=0)


    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))


    # 如果存在预训练权重则载入
    model = densenet121(num_classes=args.num_classes).to(device)
    #if os.path.exists(args.weights):
        #load_state_dict(model, args.weights)

    model_weight_path = "./DenseNet121_bs=100八分类.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)


    print(model.buffers)
    # validate
    acc = evaluate(model=model,
                       data_loader=test_loader,
                       device=device)

    print("accuracy: {}".format(round(acc, 3)))
    print('acc1 = {}, acc2 = {}'.format(acc, round(acc, 3)))

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t' % (
        round(acc, 3)))  #  四分类Best_ACC: 0.984/ 0.976

    print('Finished Training')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--weights', type=str, default='DenseNet121_bs=100八分类.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
