import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss', 'dice_loss']

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    B, c, H, W, Z = output.size()
    for i in range(c):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, c, H, W, Z = output.size()
    for i in range(c):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]

        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss
            
def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, c, H, W, Z = output.size()
    for i in range(c):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target == 4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3


def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    #logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum, [loss1.data, loss2.data, loss3.data]


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
from skimage.measure import label
import numpy as np
def cnh_loss(output, num_cls=4, eps=1e-6):
    """
    output: (B, C, D, H, W)  已经 softmax
    """
    B, C, D, H, W = output.size()
    loss = 0.0

    for b in range(B):
        probs = output[b:b+1]  # (1,C,D,H,W)

        pred_labels = torch.argmax(probs, dim=1)  # (1,D,H,W)
        binary_mask = (pred_labels != 0).float()

        mask_np = binary_mask[0].detach().cpu().numpy().astype(np.int32)
        labeled_mask_np, num_regions = label(mask_np, connectivity=3, return_num=True)

        if num_regions == 0:
            continue

        all_regions = []
        region_sizes = []

        for region_id in range(1, num_regions + 1):
            coords = np.where(labeled_mask_np == region_id)

            coords_tensor = tuple(
                torch.tensor(c, device=output.device, dtype=torch.long)
                for c in coords
            )

            labels_in_region = pred_labels[0][coords_tensor]
            probs_in_region = probs[0, labels_in_region,
                                    coords_tensor[0],
                                    coords_tensor[1],
                                    coords_tensor[2]]

            avg_prob = torch.mean(probs_in_region)
            size = probs_in_region.numel()

            region_sizes.append(size)
            all_regions.append({
                'avg_prob': avg_prob,
                'size': size
            })

        total_size = sum(region_sizes)
        largest_size = max(region_sizes)
        largest_ratio = largest_size / (total_size + eps)
        alpha = 0.01 + 0.06 * largest_ratio

        # 计算 credibility
        for r in all_regions:
            r['cred'] = r['avg_prob'] * (r['size'] ** alpha)

        # 找最大
        center_region = max(all_regions, key=lambda x: x['cred'])

        other_cred = 0.0
        for r in all_regions:
            if r is not center_region:
                other_cred += r['cred']

        center_size = center_region['size']

        loss_b = (1.0 - center_size / (total_size + eps)) * other_cred
        loss += loss_b

    return loss / B

def ih_loss(output, window_size=3, eps=1e-6):

    B, C, D, H, W = output.size()

    bg = output[:, 0:1]          # (B,1,D,H,W)
    fg = output[:, 1:]           # (B,3,D,H,W)

    fg_max, _ = torch.max(fg, dim=1, keepdim=True)


    diff = torch.relu(bg - fg_max)


    k = window_size
    pad = k // 2
    kernel = torch.ones((1, 1, k, k, k), device=output.device)

    diff_sum = F.conv3d(diff, kernel, padding=pad)

    loss = torch.mean(diff_sum)

    return loss
