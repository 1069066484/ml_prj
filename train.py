# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: codes of model training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import os
import argparse
from Fer2013Dataset_torch import Fer2013Dataset
from torch.autograd import Variable
import VGG_torch as VGG
from Lglobal_defs import *
from Ldata_helper import *
from LeNet5 import LeNet5
import utils




def calcl_mat(pred, label, m=None):
    if m is None:
        m = np.zeros([7,7])
    for p,l in zip(pred,label):
        m[p][l] += 1
    return m


batch_size = 64
lr = 0.01
resume = False

use_cuda = torch.cuda.is_available()
print('use_cuda=',use_cuda)
acc_pubT_best = 0
acc_priT_best = 0
acc_pubT_best_epoch = 0
acc_priT_best_epoch = 0

start_epoch = 0

lr_dec_start = 80
lr_dec_every = 5
lr_dec_rate = 0.9

cut_size = 44
total_epoch = 250




# tr_loss train_acc    pubT_loss pubT_acc      priT_loss priT_acc
hists = [[],[],[],[],[],[]]


trans_tr = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])
trans_te = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops]))
    ]) 

training_set = Fer2013Dataset(split='Training', transform=trans_tr)
tr_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
pubT_set = Fer2013Dataset(split='pubT', transform=trans_te)
pubT_loader = DataLoader(pubT_set, batch_size=batch_size, shuffle=False, num_workers=0)
priT_set = Fer2013Dataset(split='priT', transform=trans_te)
priT_loader = DataLoader(priT_set, batch_size=batch_size, shuffle=False, num_workers=0)

# net_name = 'VGG19'
net_name = 'LeNet5_cv1'
net = LeNet5()
# net = VGG.VGG(VGG.VGGType.VGG19)
path = mk_dir(join(PATH_SAVING, 'training_log', net_name))


if resume:
    checkpoint = torch.load(join(path, 'priT_model.t7'))
    net.load_state_dict(checkpoint['net'])
    acc_pubT_best = checkpoint['acc_pubT_best']
    acc_priT_best = checkpoint['acc_priT_best']

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

class_mat = [None,None,None]

def train(epoch):
    global acc, acc_train
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if epoch > lr_dec_start and lr_dec_start >= 0:
        frac = (epoch - lr_dec_start) // lr_dec_every
        dec_frac = lr_dec_rate ** frac
        curr_lr = lr * dec_frac
        utils.set_lr(optimizer, curr_lr)
    else:
        curr_lr = lr
    # print(1)
    for batch_idx, (inputs, targets) in enumerate(tr_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += float(loss.item())
        _, pred = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += pred.eq(targets.data).cpu().sum()
        # print(2)
        if epoch == total_epoch - 1:
            class_mat[0] = calcl_mat(pred.cpu().numpy(), targets.data.cpu().numpy(), class_mat[0])
    print("training... batch:{}   loss={}    acc={}%({}/{})".format(batch_idx, train_loss/(batch_idx+1), 100.*correct/total,correct,total))
    acc_train = 100.*correct / total
    hists[0].append(train_loss/(batch_idx+1))
    hists[1].append(acc_train)


def pub_test(epoch):
    global acc_pubT_best, acc_pubT, acc_pubT_best_epoch
    net.eval()
    pubT_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(pubT_loader):
        batch, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
        target = Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(batch, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        pubT_loss += float(loss.item())
        _, pred = torch.max(outputs_avg.data, 1)
        if epoch == total_epoch - 1:
            class_mat[1] = calcl_mat(pred.cpu().numpy(), targets.data.cpu().numpy(), class_mat[1])
        total += targets.size(0)
        correct += pred.eq(targets.data).cpu().sum()
    print("pub_test... batch:{}   loss={}    acc={}%({}/{})".format(batch_idx, pubT_loss/(batch_idx+1), 100.*correct/total,correct,total))

    acc_pubT = 100.*correct/total
    hists[2].append(pubT_loss/(batch_idx+1))
    hists[3].append(acc_pubT)
    if acc_pubT > acc_pubT_best:
        acc_pubT_best_epoch = epoch
        print('Best public test acc:', acc_pubT)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': acc_pubT,
            'epoch': epoch
            }
        torch.save(state, join(path, 'pubT_model.t7'))
        acc_pubT_best = acc_pubT

    
def pri_test(epoch):
    global acc_priT_best, acc_priT, acc_priT_best_epoch
    net.eval()
    priT_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(priT_loader):
        batch, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
        target = Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(batch, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        priT_loss += float(loss.item())
        _, pred = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += pred.eq(targets.data).cpu().sum()
        if epoch == total_epoch - 1:
            class_mat[2] = calcl_mat(pred.cpu().numpy(), targets.data.cpu().numpy(), class_mat[2])
    print("pri_test... batch:{}   loss={}    acc={}%({}/{})".format(batch_idx, priT_loss/(batch_idx+1), 100.*correct/total,correct,total))

    acc_priT = 100.*correct/total
    hists[4].append(priT_loss/(batch_idx+1))
    hists[5].append(acc_priT)
    if acc_priT > acc_priT_best:
        acc_priT_best_epoch = epoch
        print('Best public test acc:', acc_pubT)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc_priT_best': acc_priT_best,
            'acc_pubT_best_epoch':acc_pubT_best_epoch,
            'acc_priT_best_epoch': acc_priT_best_epoch
            }
        torch.save(state, join(path, 'priT_model.t7'))
        acc_priT_best = acc_pubT


use_modelarts = True
try:
    from modelarts.session import Session
except:
    use_modelarts = False




def save():
    np.save(join(path, npfn('hist')), np.array(hists))
    if class_mat[0] is not None:
        np.save(join(path, npfn('class_mat')), np.array(class_mat))
    if use_modelarts:
        session.upload_data(bucket_path="obs-optimusling2/ml_prj/Lsaving/train_log/" + net_name, path=path)


if __name__=='__main__':
    if use_modelarts:
        session = Session()
    for epoch in range(start_epoch, total_epoch):
        print("\n\nepoch=", epoch)
        train(epoch)
        pub_test(epoch)
        pri_test(epoch)
        if epoch % 4 == 0:
            save()
    save()

    print("acc_pubT_best: %0.3f    epoch: %d" % (acc_pubT_best, acc_pubT_best_epoch))
    print("acc_priT_best: %0.3f    epoch: %d" % (acc_priT_best, acc_priT_best_epoch))





