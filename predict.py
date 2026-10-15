import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc
from medpy.metric import hd95

cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot

patch_size = 80

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1),
                               torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))


from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure


def compute_surface_points(mask, spacing=(1.0, 1.0, 1.0)):
    """
    提取mask的表面点（边界点）

    Args:
        mask: 二值mask，shape (H, W, D)
        spacing: 体素间距 (z, y, x) 或 (x, y, z)，单位mm

    Returns:
        surface_points: 表面点坐标 (N, 3)，单位为物理坐标(mm)
    """
    # 生成3D结构元素（6邻域）
    struct = generate_binary_structure(3, 1)

    # 腐蚀得到内部区域
    eroded = binary_erosion(mask, structure=struct)

    # 表面 = 原区域 - 腐蚀区域
    surface = mask & (~eroded)

    # 获取表面点坐标
    points = np.argwhere(surface)

    # 转换为物理坐标（乘以体素间距）
    points_physical = points * spacing

    return points_physical


def compute_asd(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    """
    计算平均表面距离 (Average Surface Distance, ASD)

    Args:
        mask_gt: 真实mask，二值数组，shape (H, W, D)
        mask_pred: 预测mask，二值数组，shape (H, W, D)
        spacing: 体素间距 (z, y, x) 或 (x, y, z)，单位mm

    Returns:
        asd: 平均表面距离 (mm)
    """
    # 提取表面点
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    # 如果没有表面点，返回一个大的数值
    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0  # 一个表面缺失，返回较大值

    # 构建KD树加速最近邻搜索
    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    # 从GT到Pred的距离
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)

    # 从Pred到GT的距离
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    # 平均对称表面距离
    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    asd = np.mean(all_distances)

    return asd


def compute_asd_percentile(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0), percentile=95):
    """
    计算百分位数表面距离（如95%百分位，类似HD95但使用平均而非最大）

    Args:
        mask_gt: 真实mask
        mask_pred: 预测mask
        spacing: 体素间距
        percentile: 百分位数（默认95）

    Returns:
        percentile_distance: 百分位数表面距离
    """
    # 提取表面点
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    # 计算两个方向的距离
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    # 计算百分位数
    percentile_dist = np.percentile(all_distances, percentile)

    return percentile_dist


def compute_asd_symmetric(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    """
    计算对称平均表面距离 (Symmetric Average Surface Distance)
    等同于ASD，只是明确表示对称性

    Args:
        mask_gt: 真实mask
        mask_pred: 预测mask
        spacing: 体素间距

    Returns:
        asd_sym: 对称平均表面距离 (mm)
    """
    return compute_asd(mask_gt, mask_pred, spacing)


def compute_surface_metrics(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    """
    计算完整的表面距离指标集合

    Args:
        mask_gt: 真实mask
        mask_pred: 预测mask
        spacing: 体素间距

    Returns:
        dict: 包含ASD, HD95, ASD_95等指标
    """
    # 提取表面点
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return {'ASD': 0.0, 'HD95': 0.0, 'ASD_95': 0.0}
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return {'ASD': 100.0, 'HD95': 100.0, 'ASD_95': 100.0}

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    # 计算两个方向的距离
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    # 计算各种指标
    asd = np.mean(all_distances)
    hd95 = np.percentile(all_distances, 95)
    asd_95 = np.percentile(all_distances, 95)  # 实际上和HD95一样

    return {
        'ASD': asd,
        'HD95': hd95,
        'ASD_95': asd_95
    }


# 修改您的cal_hd95函数，添加ASD计算
def cal_hd95(output, target, spacing=(1.0, 1.0, 1.0)):
    """
    计算HD95和ASD指标
    spacing: 体素间距 (z, y, x)，通常从数据集中获取
    """
    # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole, asd_whole = compute_metrics_with_spacing(mask_gt, mask_pred, spacing)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = ((target == 1) | (target == 3)).astype(int)
    mask_pred = ((output == 1) | (output == 3)).astype(int)
    hd95_core, asd_core = compute_metrics_with_spacing(mask_gt, mask_pred, spacing)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh, asd_enh = compute_metrics_with_spacing(mask_gt, mask_pred, spacing)
    del mask_gt, mask_pred

    # enhancing post-processing (if needed)
    mask_gt = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
        mask_pred = (output == 3).astype(int) * 0
    else:
        mask_pred = (output == 3).astype(int)
    hd95_enhpro, asd_enhpro = compute_metrics_with_spacing(mask_gt, mask_pred, spacing)
    del mask_gt, mask_pred

    return (hd95_whole, hd95_core, hd95_enh, hd95_enhpro), (asd_whole, asd_core, asd_enh, asd_enhpro)


def compute_metrics_with_spacing(mask_gt, mask_pred, spacing):
    """
    计算HD95和ASD
    """
    # 检查是否为空
    if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
        return 0.0, 0.0
    elif np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return 100.0, 100.0

    # 提取表面点
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0, 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0, 100.0

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    # 计算两个方向的距离
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    # 计算HD95 (95th percentile)
    hd95 = np.percentile(all_distances, 95)

    # 计算ASD (mean)
    asd = np.mean(all_distances)

    return hd95, asd

def softmax_output_miou_class4(output, target):
    eps = 1e-8

    def compute_iou(o, t):
        intersect = torch.sum(o * t, dim=(1,2,3))
        union = torch.sum(o, dim=(1,2,3)) + torch.sum(t, dim=(1,2,3)) - intersect
        return (intersect + eps) / (union + eps)

    # === 三个子区域 ===
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    iou_ncr = compute_iou(o1, t1)

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    iou_edema = compute_iou(o2, t2)

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    iou_enhancing = compute_iou(o3, t3)

    # === whole tumor ===
    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    iou_whole = compute_iou(o_whole, t_whole)

    # === tumor core ===
    o_core = o1 + o3
    t_core = t1 + t3
    iou_core = compute_iou(o_core, t_core)

    # === mIoU（平均）===
    miou = (iou_ncr + iou_edema + iou_enhancing) / 3.0

    # 和你dice_evaluate对齐
    iou_evaluate = torch.cat((
        iou_whole.unsqueeze(1),
        iou_core.unsqueeze(1),
        iou_enhancing.unsqueeze(1),
        iou_ncr.unsqueeze(1),
        iou_edema.unsqueeze(1),
        iou_enhancing.unsqueeze(1),
        miou.unsqueeze(1)
    ), dim=1)

    return iou_evaluate.cpu().numpy()

def softmax_output_sensitivity_class4(output, target):
    eps = 1e-8

    def compute_sens(o, t):
        tp = torch.sum(o * t, dim=(1,2,3))
        fn = torch.sum((1 - o) * t, dim=(1,2,3))
        return (tp + eps) / (tp + fn + eps)

    # === 三个子区域 ===
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    sens_ncr = compute_sens(o1, t1)

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    sens_edema = compute_sens(o2, t2)

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    sens_enhancing = compute_sens(o3, t3)

    # === whole tumor ===
    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    sens_whole = compute_sens(o_whole, t_whole)

    # === tumor core ===
    o_core = o1 + o3
    t_core = t1 + t3
    sens_core = compute_sens(o_core, t_core)

    # === 拼接（和你 dice_evaluate 对齐）===
    sens_evaluate = torch.cat((
        sens_whole.unsqueeze(1),
        sens_core.unsqueeze(1),
        sens_enhancing.unsqueeze(1),
        sens_ncr.unsqueeze(1),
        sens_edema.unsqueeze(1),
        sens_enhancing.unsqueeze(1)
    ), dim=1)

    return sens_evaluate.cpu().numpy()

def test_softmax(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None):

    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_asd_evaluation = AverageMeter()
    vals_miou = AverageMeter()
    vals_sens = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    else:
        # BraTS2020 / MET / PED (4-class: BG + NCR/ED/ET)
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro','ncr_net', 'edema', 'enhancing'
        class_separate = 'ncr_net', 'edema', 'enhancing'
        

    for i, data in enumerate(test_loader):

        target = data[1].cuda()
        x = data[0].cuda()

        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int_(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int_(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = np.int_(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int_(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = np.int_(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int_(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)
        # weight2 = weight1.repeat(len(names), 2, 1, 1, 1)
        # nvi

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        # pred_edema = torch.zeros(len(names), 2, H, W, Z).float().cuda()
        # pred_ncr = torch.zeros(len(names), 2, H, W, Z).float().cuda()
        # pred_enhance = torch.zeros(len(names), 2, H, W, Z).float().cuda()
        # pred_bg = torch.zeros(len(names), 2, H, W, Z).float().cuda()
        model.module.is_training=False
        # rou_edemas = torch.zeros(1,4).float().cuda()
        # rou_ncrs = torch.zeros(1, 4).float().cuda()
        # rou_enhances = torch.zeros(1,4).float().cuda()
        # rou_bgs = torch.zeros(1,4).float().cuda()
        # j = 0
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    pred_part = model(x_input,None,None,mask)


                    # rou_edemas += rou_edema
                    # rou_ncrs += rou_ncr
                    # rou_enhances += rou_enhance
                    # rou_bgs += rou_bg
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
                    # pred_edema[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] = bi_outputs[2]
                    # pred_ncr[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] = bi_outputs[1]
                    # pred_enhance[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] = bi_outputs[3]
                    # pred_bg[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] = bi_outputs[0]
                    # j = j + 1
        pred = pred / weight
        # entropy = -torch.sum(pred * torch.log(pred + 1e-12), dim=1)
        # pred_edema = pred_edema / weight2
        # pred_ncr = pred_ncr / weight2
        # pred_enhance = pred_enhance / weight2
        # pred_bg = pred_bg / weight2
        # edema_entropy = -torch.sum(pred_edema * torch.log(pred_edema + 1e-12), dim=1)
        # ncr_entropy = -torch.sum(pred_ncr * torch.log(pred_ncr + 1e-12), dim=1)
        # enhance_entropy = -torch.sum(pred_enhance * torch.log(pred_bg + 1e-12), dim=1)
        # bg_entropy = -torch.sum(pred_bg * torch.log(pred_bg + 1e-12), dim=1)
        # # rou_edemas = rou_edemas / j
        # # rou_ncrs = rou_ncrs / j
        # # rou_enhances = rou_enhances / j
        # # rou_bgs = rou_bgs / j
        # # print(rou_edemas, rou_ncrs, rou_enhances, rou_bgs)
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)
        # pred_edema = torch.argmax(pred_edema, dim=1)
        # pred_ncr = torch.argmax(pred_ncr, dim=1)
        # pred_enhance = torch.argmax(pred_enhance, dim=1)
        # pred_bg = torch.argmax(pred_bg, dim=1)

        if dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        else:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            scores_miou = softmax_output_miou_class4(pred, target)
            scores_sens = softmax_output_sensitivity_class4(pred, target)
            try:
                spacing = test_loader.dataset.spacing
            except Exception:
                spacing = (1.0, 1.0, 1.0)
            scores_hd95, scores_asd = cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy(), spacing)
            scores_hd95 = np.array(scores_hd95)
            scores_asd = np.array(scores_asd)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            vals_miou.update(scores_miou[k])
            vals_hd95_evaluation.update(scores_hd95)
            vals_asd_evaluation.update(scores_asd)
            vals_sens.update(scores_sens[k])

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            msg += ', HD95: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
            msg += ', ASD: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_asd)])
            msg += ', mIoU: '
            msg += ', '.join(['{:.4f}'.format(v) for v in scores_miou[k]])
            msg += ', Sensitivity: '
            msg += ', '.join(['{:.4f}'.format(v) for v in scores_sens[k]])

            logging.info(msg)
        # import nibabel as nib
        # import os
        #
        # # === 保存预测结果为 .nii.gz 文件 ===
        # save_dir1 = '/data2/xy/mmformer/output/vision/moegw_ori_ped_ada_pred'
        # os.makedirs(save_dir1, exist_ok=True)
        # # save_dir2 = '/data2/xy/mmformer/output/vision/moe3_bi_entropy'
        # # os.makedirs(save_dir2, exist_ok=True)
        # # #
        # for k, name in enumerate(names):
        # #     # pred[k]: [H, W, Z]
        # #     # pred_edema_np = pred_edema[k].detach().cpu().numpy().astype(np.uint8)
        # #     # pred_ncr_np = pred_ncr[k].detach().cpu().numpy().astype(np.uint8)
        # #     # pred_enhance_np = pred_enhance[k].detach().cpu().numpy().astype(np.uint8)
        # #     # pred_bg_np = pred_bg[k].detach().cpu().numpy().astype(np.uint8)
        # #     # entropy_np = entropy[k].detach().cpu().numpy().astype(np.float32)
        #     pred_np = pred[k].detach().cpu().numpy().astype(np.uint8)
        # #     # entropy_edema_np = edema_entropy[k].detach().cpu().numpy().astype(np.float32)
        # #     # entropy_ncr_np = ncr_entropy[k].detach().cpu().numpy().astype(np.float32)
        # #     # entropy_bg_np = bg_entropy[k].detach().cpu().numpy().astype(np.float32)
        # #     # entropy_enhance_np = enhance_entropy[k].detach().cpu().numpy().astype(np.float32)
        # #
        #     # 获取 affine 和 header
        #     affine = getattr(test_loader.dataset, 'affine', np.eye(4))
        #     header = getattr(test_loader.dataset, 'header', None)
        # #
        # #     # pred_edema_img = nib.Nifti1Image(pred_edema_np, affine, header)
        # #     # pred_ncr_img = nib.Nifti1Image(pred_ncr_np, affine, header)
        # #     # pred_enhance_img = nib.Nifti1Image(pred_enhance_np, affine, header)
        # #     # pred_bg_img = nib.Nifti1Image(pred_bg_np, affine, header)
        # #     # save_path_edema = os.path.join(save_dir1, f"{name}_edema.nii.gz")
        # #     # save_path_ncr = os.path.join(save_dir1, f"{name}_ncr.nii.gz")
        # #     # save_path_enhance = os.path.join(save_dir1, f"{name}_enhance.nii.gz")
        # #     # save_path_bg = os.path.join(save_dir1, f"{name}_bg.nii.gz")
        # #     # nib.save(pred_edema_img, save_path_edema)
        # #     # nib.save(pred_ncr_img, save_path_ncr)
        # #     # nib.save(pred_enhance_img, save_path_enhance)
        # #     # nib.save(pred_bg_img, save_path_bg)
        # #
        # #     # entropy_edema_img = nib.Nifti1Image(entropy_edema_np, affine, header)
        # #     # entropy_ncr_img = nib.Nifti1Image(entropy_ncr_np, affine, header)
        # #     # entropy_enhance_img = nib.Nifti1Image(entropy_enhance_np, affine, header)
        # #     # entropy_bg_img = nib.Nifti1Image(entropy_bg_np, affine, header)
        # #     # save_path_edema = os.path.join(save_dir2, f"{name}_edema_entropy.nii.gz")
        # #     # save_path_ncr = os.path.join(save_dir2, f"{name}_ncr_entropy.nii.gz")
        # #     # save_path_enhance = os.path.join(save_dir2, f"{name}_enhance_entropy.nii.gz")
        # #     # save_path_bg = os.path.join(save_dir2, f"{name}_bg_entropy.nii.gz")
        # #     # nib.save(entropy_edema_img, save_path_edema)
        # #     # nib.save(entropy_ncr_img, save_path_ncr)
        # #     # nib.save(entropy_enhance_img, save_path_enhance)
        # #     # nib.save(entropy_bg_img, save_path_bg)
        # #
        #     pred_img = nib.Nifti1Image(pred_np, affine, header)
        #     save_path = os.path.join(save_dir1, f"{name}_pred.nii.gz")
        #     nib.save(pred_img, save_path)

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_asd_evaluation.avg)])
    msg += ', '.join(['{:.4f}'.format(v) for v in vals_miou.avg])
    msg += ', '.join(['{:.4f}'.format(v) for v in vals_sens.avg])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    model.train()
    return vals_evaluation.avg, vals_hd95_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

