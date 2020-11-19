# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
import time
import argparse
import shutil
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision.models import resnet

from nni.compression.torch.pruning.amc.lib.net_measure import measure_model
from nni.compression.torch.pruning.amc.lib.utils import get_output_folder
from nni.compression.torch import ModelSpeedup

from data import get_dataset
from utils import AverageMeter, accuracy, progress_bar

sys.path.append('../models')
from mobilenet import MobileNet

import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # it's same with .mean(3).mean(2), but
        # speedup only suport the mean option
        # whose output only have two dimensions
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def parse_args():
    parser = argparse.ArgumentParser(description='AMC train / fine-tune script')
    parser.add_argument('--model_type', default='mobilenet', type=str,
        choices=['mobilenet', 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50'],
        help='name of the model to train')
    parser.add_argument('--dataset', default='cifar10', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=4, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='cos', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default='./data', type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to fine tune')
    parser.add_argument('--mask_path', default=None, type=str, help='mask path for speedup')

    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--calc_flops', action='store_true', help='Calculate flops')

    return parser.parse_args()

def get_model(args):
    print('=> Building model..')

    if args.dataset == 'imagenet':
        n_class = 1000
    elif args.dataset == 'cifar10':
        n_class = 10
    else:
        raise NotImplementedError

    if args.model_type == 'mobilenet':
        net = MobileNet(n_class=n_class)
    elif args.model_type == 'mobilenetv2':
        net = MobileNetV2(n_class=n_class)
    elif args.model_type.startswith('resnet'):
        net = resnet.__dict__[args.model_type](pretrained=True)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, n_class)
    else:
        raise NotImplementedError

    if args.ckpt_path is not None:
        # the checkpoint can be state_dict exported by amc_search.py or saved by amc_train.py
        print('=> Loading checkpoint {} ..'.format(args.ckpt_path))
        net.load_state_dict(torch.load(args.ckpt_path, torch.device('cpu')))
        if args.mask_path is not None:
            SZ = 224 if args.dataset == 'imagenet' else 32
            data = torch.randn(2, 3, SZ, SZ)
            ms = ModelSpeedup(net, data, args.mask_path, torch.device('cpu'))
            ms.speedup_model()

    net.to(args.device)
    if torch.cuda.is_available() and args.n_gpu > 1 and False:
        net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
    return net

def train(epoch, train_loader, device):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)

def test(epoch, test_loader, device, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model_type,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, net, checkpoint_dir=log_dir)

def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, model, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth', '.best.pth'))
        torch.save(model, filename.replace('.pth', '.whole.pth'))

if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda') if torch.cuda.is_available() and args.n_gpu > 0 else torch.device('cpu')

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                    data_root=args.data_root)

    net = get_model(args)  # for measure

    if args.calc_flops:
        IMAGE_SIZE = 224 if args.dataset == 'imagenet' else 32
        n_flops, n_params = measure_model(net, IMAGE_SIZE, IMAGE_SIZE, args.device)
        print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M'.format(n_params / 1e6, n_flops / 1e6))
        exit(0)

    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, args.device, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model_type, args.dataset))
        train_type = 'train' if args.ckpt_path is None else 'finetune'
        log_dir = get_output_folder('./logs', '{}_{}_{}'.format(args.model_type, args.dataset, train_type))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader, args.device)
            test(epoch, val_loader, args.device)

        writer.close()
        print('=> Best top-1 acc: {}%'.format(best_acc))
