#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author (HDGN): Miguel Dominguez
@Contact: mad6384@rit.edu
@Original Author (DGCNN): Yue Wang
@File: main.py
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import HDGN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import torch.distributed as dist
import sys
from collections import deque


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_petroski.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model_petroski.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_dataset = ModelNet40(partition='train', num_points=args.num_points)
    train_loader = DataLoader(train_dataset, num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    test_loader = DataLoader(test_dataset, num_workers=0,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = 0

    #Try to load models
    if args.model == 'hdgn':
        model = HDGN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    sys.stdout.flush()
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze(dim=-1)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        sys.stdout.flush()

        ####################
        # Test
        ####################
        
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze(dim=-1)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits,wsl = model(data)
            loss = criterion(logits, label) + wsl
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        mean_test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              mean_test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        sys.stdout.flush()
        if mean_test_acc >= best_test_acc:
            best_test_acc = mean_test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = HDGN(args).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    ind_acc = []
    ind_per_class_acc = []
    for i in range(20):
        test_pred.append([])
        test_true.append([])
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            test_pred[i].append(logits.detach().cpu().numpy())
            test_true[i].append(label.cpu().numpy())
        test_pred[i] = np.concatenate(test_pred[i])
        ind_acc.append(metrics.accuracy_score(np.concatenate(test_true[0]), np.argmax(test_pred[i],1)))
        ind_per_class_acc.append(metrics.balanced_accuracy_score(np.concatenate(test_true[0]),  np.argmax(test_pred[i],1)))
    test_true = np.concatenate(test_true[0])
    test_pred = np.stack(test_pred)
    test_pred_mean = np.mean(test_pred,axis=0)
    test_pred_mean = np.argmax(test_pred_mean,axis=1)
    test_acc = metrics.accuracy_score(test_true, test_pred_mean)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred_mean)
    #if dist.get_rank() == 0:
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    avg_ind_acc = np.mean(ind_acc)
    avg_ind_per_class_acc = np.mean(ind_per_class_acc)
    std_ind_acc = np.std(ind_acc)
    std_ind_per_class_acc = np.std(ind_per_class_acc)
    outstr = 'Test :: ind test acc: %.6f std %.6f, ind test avg acc: %.6f std %.6f'%(avg_ind_acc,std_ind_acc,avg_ind_per_class_acc,std_ind_per_class_acc)
    io.cprint(outstr)
    


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='hdgn', metavar='N',
                        choices=['hdgn'],
                        help='Model to use, [dgcnn, graphcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--num_theta', type=int, default=4,
                        help='number of theta dimensions')
    parser.add_argument('--num_phi', type=int, default=4,
                        help='number of phi dimensions')                 
    args = parser.parse_args()

    _init_()
    #if dist.get_rank() == 0:
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    #else:
    #    io = None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        #if dist.get_rank() == 0:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        #if dist.get_rank() == 0:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
