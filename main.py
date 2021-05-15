

import os
import shutil
import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from load_dataset import collate_fn, dataset,get_image_pd
from augmentation import *
from utils import PolyOptimizer,train,validate,creatdir
from models.resnet import resnet50 as model

################################
########## 参数设置模块 ##########
################################

# 参数设置
parser = argparse.ArgumentParser()
# 数据集路径
parser.add_argument('--img_root_train', type=str, default= "./垃圾分类", help='whether to img root')
# 模型及数据存储路径
parser.add_argument('--checkpoint_dir', type=str, default='./results/resnet50/', help='directory where model checkpoints are saved')
# 网络选择
parser.add_argument('--net', dest='net',type=str, default='resnet50',help='which net is chosen for training ')
# 批次
parser.add_argument('--batch_size', type=int, default=12, help='size of each image batch')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
# 衰减率
parser.add_argument("--weight_decay", default=5e-6, type=float)
# cuda设置
parser.add_argument('--cuda', type=str, default="0", help='whether to use cuda if available')
# CPU载入数据线程设置
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# 暂停设置
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
# 迭代次数
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
# 起始次数（针对resume设置）
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
# 显示结果的间隔
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
# 确认参数，并可以通过opt.xx的形式在程序中使用该参数
opt = parser.parse_args()

##############################
########## 主函数模块 ##########
##############################

if __name__ == '__main__':
    # 创建存储及日志文件
    creatdir(opt.checkpoint_dir)
    # 获取图片路径，并划分训练集、测试集
    all_pd=  get_image_pd(opt.img_root_train)
    # print(all_pd)
    # 训练集和验证集
    train_pd,val_pd = train_test_split(all_pd, test_size=0.2, random_state=53,stratify=all_pd["label"])
    # 输出数据大小
    # print(val_pd.shape)
    # 数据增强（预处理）
    data_transforms = {
        'train': Compose([
            Resize(size=(400, 400)),
            FixRandomRotate(bound='Random'),
            RandomHflip(),
            RandomVflip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            Resize(size=(400, 400)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 数据集
    data_set = {}
    # 数据增强后的训练集
    data_set['train'] = dataset(anno_pd=train_pd, transforms=data_transforms["train"])
    # 数据增强后的验证集
    data_set['val'] = dataset(anno_pd=val_pd, transforms=data_transforms["val"])
    # 载入数据集
    dataloader = {}
    # 载入数据增强后的训练集
    dataloader['train'] = torch.utils.data.DataLoader(data_set['train'], batch_size=opt.batch_size,
                                                      shuffle=True, num_workers=opt.n_cpu, collate_fn=collate_fn)
    # 载入数据增强后的验证集
    dataloader['val'] = torch.utils.data.DataLoader(data_set['val'], batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=opt.n_cpu, collate_fn=collate_fn)
    # 获取最大迭代步数
    max_step = (len(data_set['train']) // opt.batch_size) * opt.epochs
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model=model().to(device)
    # 网络模型的选择
    # model = Net().to('cuda')
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)
    # 损失函数
    criterion = CrossEntropyLoss()
    # 学习率衰减设置
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # 暂停选项
    if opt.resume:
        model.eval()
        print('resuming finetune from %s' % opt.resume)
        try:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model.load_state_dict(torch.load(opt.resume))
            model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # 最佳准确率置0
    best_precision = 0
    # 设置损失
    lowest_loss = 10000
    # 训练
    for epoch in range(opt.start_epoch, opt.epochs):
        # 训练
        acc_train, loss_train = train(dataloader['train'], model, criterion, optimizer, epoch, print_interval=opt.print_interval,filename=opt.checkpoint_dir)
        # 在日志文件中记录每个epoch的训练精度和损失
        with open(opt.checkpoint_dir + 'record.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, train_Precision: %.8f, train_Loss: %.8f\n' % (epoch, acc_train, loss_train))
        # 测试
        precision, avg_loss = validate(dataloader['val'], model, criterion, print_interval=opt.print_interval,filename=opt.checkpoint_dir)
        # 在日志文件中记录每个epoch的验证精度和损失
        with open(opt.checkpoint_dir + 'record_val.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
        # 记录最高精度与最低loss
        is_best = precision > best_precision
        is_lowest_loss = avg_loss < lowest_loss
        best_precision = max(precision, best_precision)
        lowest_loss = min(avg_loss, lowest_loss)
        print('--'*30)
        print(' * Accuray {acc:.3f}'.format(acc=precision), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss:.3f}'.format(loss=avg_loss), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        print('--' * 30)
        # 保存最新模型
        save_path = os.path.join(opt.checkpoint_dir,'checkpoint.pth')
        torch.save(model.state_dict(),save_path)
        # 保存准确率最高的模型
        best_path = os.path.join(opt.checkpoint_dir,'best_model.pth')
        if is_best:
            shutil.copyfile(save_path, best_path)
        # 保存损失最低的模型
        lowest_path = os.path.join(opt.checkpoint_dir, 'lowest_loss.pth')
        if is_lowest_loss:
            shutil.copyfile(save_path, lowest_path)
