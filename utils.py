import shutil
import torch
import time
import random
import numpy as np
import os
from torch.autograd import Variable

def creatdir(filename):
    # 创建存储文件
    if not os.path.exists(filename):
        os.makedirs(filename)
    # 创建epoch日志文件
    if not os.path.exists(filename + 'record.txt'):
        with open(filename + 'record.txt', 'w') as acc_file:
            pass


# 计算精度和时间的变化
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 计算准确率
def accuracy(output, labels, topk=(1,)):
    # 最终准确率置零
    final_acc = 0
    # 获取最大分值
    maxk = max(topk)
    # 训练总数
    general_number = labels.size(0)
    # 预测正确的个数
    pred_correct_number = 0
    # 获取结果张量中的top1分值
    prob, pred = output.topk(maxk, 1, True, True)  # out = (1:94.1,2:3.8,3:2.1)
    # 判定预测正确的数目
    for j in range(pred.size(0)):
        if int(labels[j]) == int(pred[j]):
            pred_correct_number += 1
    # 如果训练总数为0，即不合理的情况，判定准确率为0
    if general_number == 0:
        final_acc = 0
    # 否则准确率 = 预测正确的个数 / 训练总数
    else:
        final_acc = pred_correct_number / general_number
    # 返回准确率的百分值和训练总数
    return final_acc * 100, general_number

# 训练函数
def train(train_loader, model, criterion, optimizer, epoch, print_interval,filename):
    # 记录数据
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # 转换为训练模式
    model.train()
    # 时间记录
    end = time.time()
    # 从训练集迭代器中获取训练数据
    for i, (images, labels) in enumerate(train_loader):
        # 评估图片读取耗时
        data_time.update(time.time() - end)
        # 将图片和标签转化为tensor
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        images = images.to(device)
        labels = torch.from_numpy(np.array(labels)).long().to(device)
        # 梯度归零
        optimizer.zero_grad()
        # 将图片输入网络，前传，生成预测值
        output = model(images)
        # 计算loss
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))
        # 计算top1正确率
        prec, PRED_COUNT = accuracy(output.data, labels, topk=(1, 1))
        acc.update(prec, PRED_COUNT)
        # 对梯度进行反向传播，使用随机梯度下降更新网络权重
        loss.backward()
        optimizer.step()
        # 评估训练耗时
        batch_time.update(time.time() - end)
        end = time.time()
        # 打印耗时与结果
        if i % print_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))
        # 创建train iter日志文件
        if not os.path.exists(filename + 'record_iter_train.txt'):
            with open(filename + 'record_iter_train.txt', 'w') as iter_train_file:
                pass
        with open(filename + 'record_iter_train.txt', 'a') as iter_train_file:
            iter_train_file.write('train_Precision: %.8f, train_Loss: %.8f\n' % (acc.val, losses.val))
    return acc.avg,losses.avg

# 验证函数
def validate(val_loader, model, criterion, print_interval,filename):
    # 记录数据
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # 转换为验证模式
    model.eval()
    # 时间记录
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 将数据转化为tensor
        images = images.to(device)
        labels = torch.from_numpy(np.array(labels)).long().to(device)
        # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)
        # 计算损失和准确率
        prec, PRED_COUNT = accuracy(output.data, labels, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        acc.update(prec, PRED_COUNT)
        # 耗时记录
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_interval == 0:
            print('TrainVal: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
        # 创建val iter日志文件
        if not os.path.exists(filename + 'record_iter_val.txt'):
            with open(filename + 'record_iter_val.txt', 'w') as iter_val_file:
                pass
        with open(filename + 'record_iter_val.txt', 'a') as iter_val_file:
            iter_val_file.write('val_Precision: %.8f, val_Loss: %.8f\n' % (acc.val, losses.val))

    return acc.avg, losses.avg





class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1