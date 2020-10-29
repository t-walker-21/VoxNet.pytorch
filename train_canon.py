#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: train.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description:
'''

from __future__ import print_function
import argparse
import sys
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from voxnet import VoxNetSegmentation
sys.path.insert(0, './data/')
from modelnet10 import ModelNet10
import open3d as o3d
import numpy as np
from numpy import sin, cos

CLASSES = {
    0: 'bathtub',
    1: 'chair',
    2: 'dresser',
    3: 'night_stand',
    4: 'sofa',
    5: 'toilet',
    6: 'bed',
    7: 'desk',
    8: 'monitor',
    9: 'table'
}



def blue(x): return '\033[94m' + x + '\033[0m'

def show_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def voxel_to_pointcloud(voxel_grid, scale=5.0):
    """
    Convert a voxelgrid to a pointcloud
    """

    grid_size = voxel_grid.shape[0]

    center = grid_size / 2

    point_cloud = []

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):

                if voxel_grid[i][j][k] <= 0.4:
                    continue

                ptx = (center - i) * scale
                pty = (center - j) * scale
                ptz = (center - k) * scale

                point_cloud.append([ptx, pty, ptz])

    point_cloud = np.array(point_cloud)

    return point_cloud

def point_cloud_to_voxel(point_cloud, scale=5.0):
    """
    Convert pointcloud to voxel grid
    """

    voxel_size = 32
    center = voxel_size / 2

    voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size))

    scaled_pcd = point_cloud / scale

    for pt in scaled_pcd:
        
        x_coord = min(int(center + pt[0]), 31)
        y_coord = min(int(center + pt[1]), 31)
        z_coord = min(int(center + pt[2]), 31)

        voxel_grid[x_coord][y_coord][z_coord] = 1

    return voxel_grid

def rand_rotation_matrix(deflection=1.0):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random rotation. Small
    deflection => small perturbation.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    theta = np.random.uniform(0, 2.0*deflection*np.pi) # Rotation about the pole (Z).
    phi = np.random.uniform(0, 2.0*np.pi) # For direction of pole deflection.
    z = np.random.uniform(0, 2.0*deflection) # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    # Compute the row vector S = Transpose(V) * R, where R is a simple
    # rotation by theta about the z-axis.  No need to compute Sz since
    # it's just Vz.

    st = sin(theta)
    ct = cos(theta)
    Sx = Vx * ct - Vy * st
    Sy = Vx * st + Vy * ct
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which
    # is equivalent to V S - R.
    
    M = np.array((
            (
                Vx * Sx - ct,
                Vx * Sy - st,
                Vx * Vz
            ),
            (
                Vy * Sx + st,
                Vy * Sy - ct,
                Vy * Vz
            ),
            (
                Vz * Sx,
                Vz * Sy,
                1.0 - z   # This equals Vz * Vz - 1.0
            )
            )
    )
    return M


# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='data/ModelNet10', help="dataset path")
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--n-epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--model-name', type=str, default='toilet', help='model name')
opt = parser.parse_args()
# print(opt)

CLASSES = {
    0: opt.model_name
}

N_CLASSES = len(CLASSES)

# 创建目录
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 固定随机种子
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 数据加载
train_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
test_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')

train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

# VoxNet
voxnet = VoxNetSegmentation(n_classes=N_CLASSES)

print(voxnet)
#criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()

print ("init model")

# 加载权重
if opt.model != '':
    voxnet.load_state_dict(torch.load(opt.model))

# 优化器
optimizer = optim.Adam(voxnet.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
voxnet.cuda()

num_batch = len(train_dataset) / opt.batchSize
step = 0

print ("starting training")

for epoch in range(opt.n_epoch):
    # scheduler.step()
    for i, sample in enumerate(train_dataloader, 0):
        # 读数据

        voxel, cls_idx, voxel_canon = sample['voxel'], sample['cls_idx'], sample['canon_voxel']
        voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
        voxel_canon = voxel_canon.cuda().float()
        voxel = voxel.float()  # Voxel原来是int类型(0,1),需转float, torch.Size([256, 1, 32, 32, 32])
        optimizer.zero_grad()

        # 网络切换训练模型
        voxnet = voxnet.train()
        pred = voxnet(voxel)  # torch.Size([256, 10])

        loss = criterion(pred, voxel_canon)

        # 反向传播, 更新权重
        loss.backward()
        optimizer.step()

        # 计算该batch的预测准确率
        print (loss.item())

        if loss.item() < 0.08 and True:

            view_idx = np.random.randint(opt.batchSize)

            print ("GT")
            voxel_np = voxel_canon[view_idx][0].detach().cpu().numpy()
            pcd = voxel_to_pointcloud(voxel_np)
            show_pcd(pcd)

            print ("INPUT")
            voxel_np = voxel[view_idx][0].detach().cpu().numpy()
            pcd = voxel_to_pointcloud(voxel_np)
            show_pcd(pcd)

            print ("Prediction")
            voxel_np = pred[view_idx][0].detach().cpu().numpy()
            pcd = voxel_to_pointcloud(voxel_np)
            show_pcd(pcd)

        
        step += 1


        # 每5个batch进行一次test
        if i % 100 == 0:
            j, sample = next(enumerate(test_dataloader, 0))
            voxel, cls_idx = sample['voxel'], sample['cls_idx']
            voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
            voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])
            voxnet = voxnet.eval()
            pred = voxnet(voxel)
            loss = criterion(pred, voxel_canon)
            print ("test loss: ", blue(str(loss.item())))

    # 保存权重
    torch.save(voxnet.state_dict(), '%s/canon_model_%d.pth' % (opt.outf, epoch))


# 训练后, 在测试集上评估
total_correct = 0
total_testset = 0

exit()

for i, data in tqdm(enumerate(test_dataloader, 0)):
    voxel, cls_idx = data['voxel'], data['cls_idx']
    voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
    voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])

    voxnet = voxnet.eval()
    pred = voxnet(voxel)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(cls_idx.data).cpu().sum()
    total_correct += correct.item()
    total_testset += voxel.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
