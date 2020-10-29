# coding: utf-8
import os
import sys
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '../utils/')
sys.path.insert(0, './utils/')
import binvox_rw
from numpy import sin, cos

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

    voxel_size = 64
    center = voxel_size / 2

    voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size))

    scaled_pcd = point_cloud / scale

    for pt in scaled_pcd:
        
        x_coord = min(int(center + pt[0]), voxel_size - 1)
        y_coord = min(int(center + pt[1]), voxel_size - 1)
        z_coord = min(int(center + pt[2]), voxel_size - 1)

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

class ModelNet10(Dataset):
    def __init__(self, data_root, n_classes, idx2cls, split='train', voxel_size=64):
        """
        Args:
            split (str, optional): 'train' or 'test'. Defaults to 'train'.
        """
        self.data_root = data_root
        self.n_classes = n_classes
        self.samples_str = []
        self.cls2idx = {}
        self.canon_id = 15
        self.voxel_size = voxel_size

        for k, v in idx2cls.items():

            self.cls2idx.update({v: k})

            for sample_str in glob.glob(os.path.join(data_root, v, split, '*.binvox')):
                if re.match(r"[a-zA-Z]+_\d+.binvox", os.path.basename(sample_str)):
                    self.samples_str.append(sample_str)

    def __getitem__(self, idx):

        
        #idx = np.random.randint(2)
        idx = 30
        sample_name = self.samples_str[idx]
        cls_name = re.split(r"_\d+\.binvox", os.path.basename(sample_name))[0]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]


        sample_name_canon = self.samples_str[self.canon_id]
        cls_name_canon = re.split(r"_\d+\.binvox", os.path.basename(sample_name_canon))[0]
        cls_idx_canon = self.cls2idx[cls_name_canon]
        with open(sample_name_canon, 'rb') as file:
            data_canon = np.int32(binvox_rw.read_as_3d_array(file).data)
            data_canon = data_canon[np.newaxis, :]

        rotated = []
        rotated_canon = []

        cloud = voxel_to_pointcloud(data[0])
        cloud_canon = voxel_to_pointcloud(data_canon[0])

        #rot = np.eye(3)
        rot = rand_rotation_matrix()

        for point in cloud:
            rotated.append(np.dot(point, rot))

        for point in cloud_canon:
            rotated_canon.append(np.dot(point, rot))


        rotated = np.array(rotated)
        rotated_canon = np.array(rotated_canon)

        vox_rotated = point_cloud_to_voxel(rotated)
        vox_rotated_canon = point_cloud_to_voxel(rotated_canon)

        vox_rotated = vox_rotated.reshape(1, self.voxel_size, self.voxel_size, self.voxel_size)
        vox_rotated_canon = vox_rotated_canon.reshape(1, self.voxel_size, self.voxel_size, self.voxel_size)

        sample = {'voxel': vox_rotated, 'cls_idx': cls_idx, 'canon_voxel': vox_rotated_canon}

        return sample

    def __len__(self):
        return len(self.samples_str)


if __name__ == "__main__":
    idx2cls = {0: 'bathtub', 1: 'chair', 2: 'dresser', 3: 'night_stand',
               4: 'sofa', 5: 'toilet', 6: 'bed', 7: 'desk', 8: 'monitor', 9: 'table'}

    data_root = './ModelNet10'

    dataset = ModelNet10(data_root=data_root, n_classes=10, idx2cls=idx2cls, split='train')
    cnt = len(dataset)

    data, cls_idx = dataset[0]['voxel'], dataset[1]['cls_idx']
    print("length: {cnt}\nsample data: {data}\nsample cls: {cls_idx}")
