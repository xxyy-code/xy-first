import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib
import glob
join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

# mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
#                       [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
#                       [True, True, True, True]])
mask_array = np.array([[True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True],
                      [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True],
                       [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True],
                       [True, True, True, True], [True, True, True, True],
                      [True, True, True, True]])

def make_binary_mask(yo, cls_idx):
    """
    yo: [ C, H, W, D]  one-hot label
    cls_idx: int        target class index
    """
    fg = yo[cls_idx:cls_idx+1, ...]   # [B,1,H,W,D]
    bg = 1 - fg
    binary_mask = torch.cat([bg, fg], dim=0)
    return binary_mask

import numpy as np
from scipy.ndimage import binary_erosion

def extract_boundary(mask, iterations=1):
    """
    mask: Tensor or ndarray, shape [H, W, Z]
    return: boundary mask, same shape
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    mask_np = mask_np.astype(np.bool_)
    eroded = binary_erosion(mask_np, iterations=iterations)
    boundary = mask_np ^ eroded   # 或 mask_np & (~eroded)

    return torch.from_numpy(boundary.astype(np.float32))

import nibabel as nib
import numpy as np
import os

def save_nii(volume, save_path, affine=None):
    """
    volume: Tensor or ndarray, shape [H, W, Z]
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    volume = volume.astype(np.uint8)  # 边界 0/1 就够了

    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, save_path)


class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]

        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname + '_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2, 3])


    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        # 给 x 和 y 增加一个新的维度
        x, y = x[None, ...], y[None, ...]

        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)

        y = np.reshape(y, (-1))

        one_hot_targets = np.eye(self.num_cls)[y]

        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]
        # 去掉第0维的大小为1的维度
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        # 构造每个类的二值mask
        bg_mask = make_binary_mask(yo, 0)
        ncr_mask = make_binary_mask(yo, 1)
        edema_mask = make_binary_mask(yo, 2)
        enhance_mask = make_binary_mask(yo, 3)
        ncr_boundary = extract_boundary(ncr_mask[1].squeeze(0))
        edema_boundary = extract_boundary(edema_mask[1].squeeze(0))
        enhance_boundary = extract_boundary(enhance_mask[1].squeeze(0))
        #
        target = (yo, bg_mask, ncr_mask, edema_mask, enhance_mask)
        boundarys = (ncr_boundary, edema_boundary, enhance_boundary)



        mask_idx = np.random.choice(15, 1)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        return x,target,boundarys, mask, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_energy_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]

        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname + '_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2, 3])

    #
    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)

        x, y = x[None, ...], y[None, ...]

        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)

        y = np.reshape(y, (-1))

        one_hot_targets = np.eye(self.num_cls)[y]

        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)


        mask_idx = np.random.choice(15, 1)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        label = 1
        label = torch.tensor(label).long()

        return x,yo,label, mask, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        data_file_path = os.path.join(root, test_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
