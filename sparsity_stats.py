from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import models
from flops import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--data_path', type=str, default='../../datasets/data.cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./checkpoint/exp00_debug', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='ResNet56', type=str,
                    help='architecture to use')
parser.add_argument('--sr', type=float)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

####
'''
args.data_path = '../../datasets/data.cifar10'
args.save = 'checkpoint/exp02_vgg_sr1e5_thr0.01'
args.arch = 'VGG'
args.sr = 0.00001
args.threshold = 0.5 
args.epochs = 1 
args.batch_size = 128
if not os.path.exists(args.save):
    os.makedirs(args.save)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
####

args.arch = 'ResNet56'



print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.num_classes == 10:
    train_set = datasets.CIFAR10(args.data_path, train=True)
    test_set = datasets.CIFAR10(args.data_path, train=False)
else:
    train_set = datasets.CIFAR100(args.data_path, train=True)
    test_set = datasets.CIFAR100(args.data_path, train=False)

test_set.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
model = models.__dict__[args.arch](num_classes=args.num_classes)
model.cuda()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


##############pruning filter in filter without finetuning#################
if args.sr and args.threshold:
    dummy_input = torch.randn((1, 3, 32, 32)).cuda()
    import pdb; pdb.set_trace()
    torch.onnx.export(model, dummy_input, os.path.join(args.save, 'dense.onnx'))
    model.load_state_dict(torch.load(os.path.join(args.save, 'best.pth.tar')))
    print('**** dense stats ****')
    test()
    print_model_param_nums(model)
    count_model_param_flops(model)
    model.prune(args.threshold)
    #torch.onnx.export(model, dummy_input, os.path.join(args.save, 'sparse.onnx'))
    print('########################')
    print('**** sparse stats ****')
    test()
    #print(model)
    #torch.save(model.state_dict(), os.path.join(args.save, 'pruned.pth.tar'))
    print('**** pruned ****')
    print_model_param_nums(model)
    count_model_param_flops(model)
    print('**************')
#########################################################
