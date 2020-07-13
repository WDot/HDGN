#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author (HDGN): Miguel Dominguez
@Contact: mad6384@rit.edu
@Original Author (DGCNN): Yue Wang
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    with open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', partition + '_files.txt'),'r') as f:
        files = f.readlines()
    for h5_name in files:
        f = h5py.File(h5_name.rstrip(),'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((batch_pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        batch_pc[drop_idx,:] = batch_pc[0,:] # set to the first point
    return batch_pc

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train',angle=None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.angle = angle

    def __getitem__(self, item):
        choices = np.random.choice(np.arange(self.data[item].shape[0]),self.num_points,replace=False)
        pointcloud = self.data[item][choices]
        label = self.label[item]
        if self.partition == 'train' or self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = random_point_dropout(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud.astype(np.float32), label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
