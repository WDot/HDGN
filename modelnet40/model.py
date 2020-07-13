#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author (HDGN): Miguel Dominguez
@Contact: mad6384@rit.edu
@Original Author (DGCNN): Yue Wang
@File: model.py
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #print('PAIRWISE DISTANCE SHAPE {0}'.format(pairwise_distance.shape))
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
    
def get_graph_feature(x, k=20, idx=None, concat=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base.type('torch.cuda.LongTensor')

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    if concat:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    else:
        feature = feature.permute(0,3,1,2)
  
    return feature
    
def pairwise_edge_vector(point_cloud, nn_idx):
    #BxNxF => BxNxNxF
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
      nn_idx: tensor (batch_size, num_points, K) #Only do pairwise edges with K nearest neighbors

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    num_features = point_cloud.shape[2]
    K = nn_idx.shape[2]
    point_cloud_4d_vertical = torch.unsqueeze(point_cloud,dim=2) #BxNx1xF
    cols = torch.reshape(nn_idx,[-1]) #BNK
    #rows = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_points),axis=1),[batch_size,K]),[-1])
    batch = torch.reshape(torch.unsqueeze(torch.arange(batch_size),dim=0).repeat([num_points*K,1]).t(),[-1])
    point_cloud_neighbors = point_cloud[batch,cols,:]#tf.gather_nd(point_cloud,tf.stack((batch,cols),axis=1)) #BNKxF
    #print(point_cloud_neighbors.shape)
    point_cloud_4d_horizontal = torch.reshape(point_cloud_neighbors,[batch_size,num_points,K,num_features]) #BxNxKxF
    vector = point_cloud_4d_vertical - point_cloud_4d_horizontal #BxNxKxF
    return vector    
    
def pairwise_angle(angles,standardAngles,nn_idx, numTheta,numPhi):
    #angles = BxNxKxF
    #standardAngles = MxF
    #nn_idx = BxNxK
    #pi = tf.constant(np.pi)
    batch_size = angles.shape[0]
    num_points = angles.shape[1]
    K = angles.shape[2]
    num_features = angles.shape[3]
    num_dir_neighbors = standardAngles.shape[0]
    #angles3d = tf.reshape(angles,[batch_size,num_points*K,num_features]) #BxNKxF
    #Tiling is probably cheap since the number of standard angles will likely never become very large
    #standardAngles3d = tf.tile(tf.expand_dims(standardAngles,axis=0),[batch_size,1,1])
    angles5d = torch.unsqueeze(angles,dim=3) #BxNxKx1xF
    standardAngles5d = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(standardAngles,0),0),0) #1x1x1xMxF
    elementwise_distance = torch.pow(angles5d - standardAngles5d,2) #BxNxKxMxF
    vector_distance = torch.sum(elementwise_distance,dim=4) #BxNxKxM
    vector_distance_threshold = vector_distance
    optimum_indices_encoded = torch.argmax(-vector_distance,dim=2) #BxNxM
    cols = torch.reshape(optimum_indices_encoded,[-1]) #BNM
    rows = torch.reshape(torch.unsqueeze(torch.arange(num_points),dim=1).repeat([batch_size,num_dir_neighbors]),[-1]).long()
    batch = torch.reshape(torch.unsqueeze(torch.arange(batch_size),axis=0).repeat([num_points*num_dir_neighbors,1]).t(),[-1]).long()
    optimum_indices_1d = nn_idx[batch,rows,cols]#tf.gather_nd(nn_idx,tf.stack((batch,rows,cols),axis=1))
    optimum_indices = torch.reshape(optimum_indices_1d,[batch_size,num_points,num_dir_neighbors])
    
    return optimum_indices
    
def toSpherical4D(V):
    #BxNxKxF
    pi = torch.tensor(np.pi,dtype=torch.float32)
    r = torch.sqrt(torch.sum(torch.pow(V,2),3))
    X = V[:,:,:,0]#tf.slice(V, [0,0,0,0], [-1,-1,-1,1])
    Y = V[:,:,:,1]#tf.slice(V, [0,0,0,1], [-1,-1,-1,1])
    Z = V[:,:,:,2]#tf.slice(V, [0,0,0,2], [-1,-1,-1,1])
    #print(X.shape)
    #print(Y.shape)
    #print(Z.shape)
    #print(r.shape)
    theta = torch.atan2(Y,X) + pi #Not sure if adding pi to go from -pi to pi to 0 to 2pi is correct #Bx
    #for p in sorted(V[:,2]/(r + np.finfo(np.float32).eps)):
    #    print(p)
    epsilon = torch.tensor(np.finfo(np.float32).eps,dtype=torch.float32)
    phi = torch.acos(Z/(r + epsilon))
    #print(theta.shape)
    #print(phi.shape)

    return (r,theta,phi)
    
def directional_knn(point_cloud,features,numTheta,numPhi,K,device):
    point_cloud = point_cloud.permute(0,2,1)
    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    num_features = point_cloud.shape[2]
    #distances = pairwise_distance(point_cloud)
    #nn_idx = knn(distances, k=K)
    nn_idx = knn(features, K)
    #nn_idx = nn_idx[:,:,int(K/2):]
    #nn_idx = nn_idx[:,:,::2]
    #point_cloud = point_cloud.permute(0,2,1)
    angleVectorList = init_spherical_midpoints(numTheta,numPhi,device)
    vectors = pairwise_edge_vector(point_cloud, nn_idx)
    r,theta,phi = toSpherical4D(vectors)
    #print('THETA SHAPE {0}'.format(theta.shape))
    angles = torch.stack([theta,phi],dim=3)
    directionalknn_idx = pairwise_angle(angles,angleVectorList,nn_idx,numTheta,numPhi)
    return directionalknn_idx
    
def pairwise_dist2(A, B,device):  
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
    A,    [b,m,d] matrix
    B,    [b,n,d] matrix
    Returns:
    D,    [b,m,n] matrix of pairwise distances
    """
    # squared norms of each row in A and B
    m = A.shape[1]
    n = B.shape[1]
    na = torch.sum(torch.pow(A,2), 2) #[b,m]
    nb = torch.sum(torch.pow(B,2), 2) #[b,n]

    # na as a row and nb as a co"lumn vectors
    na = torch.reshape(na, [-1,m, 1]) #BxMx1
    nb = torch.reshape(nb, [-1,1, n]) #Bx1xN

    # return pairwise euclidean difference matrix
    D = torch.max(na - 2*torch.matmul(A, B.permute([0,2,1])) + nb, torch.tensor(0.0,dtype=torch.float32,device=device)) #BxMx1 BxMxD x BxDxN Bx1xN
    return D
    
def init_spherical_midpoints(numTheta,numPhi,device):
    # 0 1*2pi/4 2*2pi/4 3*2pi/4 4*2pi/4
    pi = torch.tensor(np.pi,dtype=torch.float32,device=device)
    thetaRange = torch.arange(numTheta+1, dtype=torch.float32,device=device)
    phiRange = torch.arange(numPhi+1, dtype=torch.float32,device=device)
    thetaLeft = thetaRange[0:-1]*2.*pi/numTheta
    thetaRight = thetaRange[1:]*2.*pi/numTheta
    thetaMid = (thetaRight + thetaLeft)/2. #NT
    thetaMid2d = torch.unsqueeze(thetaMid,dim=0) #1xNT
    thetaMidTile = thetaMid2d.repeat([numPhi,1]) #NPxNT
    
    phiLeft = phiRange[0:-1]*pi/numPhi
    phiRight = phiRange[1:]*pi/numPhi
    phiMid = (phiRight + phiLeft)/2. #NP
    phiMid2d = torch.unsqueeze(phiMid,dim=0) #1xNP
    phiMidTile = phiMid2d.repeat([numTheta,1])#NTxNP
    phiMidTileTranspose = torch.t(phiMidTile) #NPxNT
    
    angleVector = torch.stack((thetaMidTile,phiMidTileTranspose),dim=2) #NPxNTx2
    angleVectorList = torch.reshape(angleVector,[-1,2]) #NPNTx2
    
    return angleVectorList
        
class HDGN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(HDGN, self).__init__()
        self.args = args
        self.k = args.k
        self.NUM_THETA = args.num_theta
        self.NUM_PHI = args.num_phi
        self.device = args.device
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.bn1b = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(512)
        #self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        


        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 128, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(64, 128, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(128),
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv1b = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 128, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn1b,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(64, 128, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(128),
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((3 + 128)*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 256, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(128, 256, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv2b = nn.Sequential(nn.Conv2d(256*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 256, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn2b,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(128, 256, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d((3 + 128 + 256)*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(256, 512, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(256, 512, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(512),
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv3b = nn.Sequential(nn.Conv2d(512*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(256, 512, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   self.bn3b,
                                   nn.LeakyReLU(negative_slope=0.2))#,
        #                           nn.Conv2d(256, 512, kernel_size=1, bias=False),
        #                           nn.BatchNorm2d(512),
        #                           nn.LeakyReLU(negative_slope=0.2))
        '''
        self.conv4 = nn.Sequential(nn.Conv2d(256*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 256, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4b = nn.Sequential(nn.Conv2d(256*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 256, kernel_size=[1, self.NUM_THETA*self.NUM_PHI], bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        '''
        #self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
        self.conv5 = nn.Sequential(nn.Conv1d((3 + 128 + 256 + 512), args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:,0:3,:]
        
        x0 = torch.unsqueeze(x,3)

        nn_idx = directional_knn(xyz,x,self.NUM_THETA,self.NUM_PHI,self.k,self.device)
        x = get_graph_feature(x, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x1 = self.conv1(x)
        x = get_graph_feature(x1, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x1b = self.conv1b(x)
        
        x1 = torch.cat((x0,x1b),dim=1)
        
        #xyz, x1pool, _ = stride_pool_module(xyz,x1,nn_idx,self.NUM_THETA*self.NUM_PHI,int(num_points/4))
        nn_idx = directional_knn(xyz,torch.squeeze(x1,3),self.NUM_THETA,self.NUM_PHI,self.k,self.device)
        x = get_graph_feature(x1, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x2 = self.conv2(x)
        x = get_graph_feature(x2, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x2b = self.conv2b(x)
        
        x2 = torch.cat((x1,x2b),dim=1)
        
        #xyz, x2pool, _ = stride_pool_module(xyz,x2,nn_idx,self.NUM_THETA*self.NUM_PHI,int(num_points/16))
        nn_idx = directional_knn(xyz,torch.squeeze(x2,3),self.NUM_THETA,self.NUM_PHI,self.k,self.device)
        x = get_graph_feature(x2, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x3 = self.conv3(x)
        x = get_graph_feature(x3, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        x3b = self.conv3b(x)
        
        x3 = torch.cat((x2,x3b),dim=1)

        #xyz, x3pool, _ = stride_pool_module(xyz,x3,nn_idx,self.NUM_THETA*self.NUM_PHI,int(num_points/8))
        #nn_idx = directional_knn(xyz,x3pool,self.NUM_THETA,self.NUM_PHI,self.k,self.device)
        #x = get_graph_feature(x3pool, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        #x4 = self.conv4(x)
        #x = get_graph_feature(x4, k=self.NUM_THETA*self.NUM_PHI, idx=nn_idx)
        #x4 = self.conv3b(x)
        #x = torch.cat((x1, x2, x3), dim=1)
        
       # x = torch.squeeze(x,3)
        x = self.conv5(torch.squeeze(x3,3))
        #x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        #x = torch.cat((torch.max(x1,dim=2)[0], torch.max(x2,dim=2)[0], torch.max(x3,dim=2)[0]), dim=1)
        
        #x = torch.squeeze(x4,3)

        #x = torch.squeeze(self.conv5(x),dim=2)
        #x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        #x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        #x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x