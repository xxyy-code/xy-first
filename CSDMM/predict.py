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
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
        o4 = o3 * 0.0
    else:
        o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1),
                               torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1),
                               torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1),
                               torch.unsqueeze(enhancing_dice, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1, 2, 3)) + eps
    denominator5 = torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t5, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1),
                               torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1),
                               torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

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
    struct = generate_binary_structure(3, 1)

    eroded = binary_erosion(mask, structure=struct)

    surface = mask & (~eroded)

    points = np.argwhere(surface)

    points_physical = points * spacing

    return points_physical


def compute_asd(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0
    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)

    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    asd = np.mean(all_distances)

    return asd


def compute_asd_percentile(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0), percentile=95):
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    percentile_dist = np.percentile(all_distances, percentile)

    return percentile_dist


def compute_asd_symmetric(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    return compute_asd(mask_gt, mask_pred, spacing)


def compute_surface_metrics(mask_gt, mask_pred, spacing=(1.0, 1.0, 1.0)):
    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return {'ASD': 0.0, 'HD95': 0.0, 'ASD_95': 0.0}
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return {'ASD': 100.0, 'HD95': 100.0, 'ASD_95': 100.0}

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    asd = np.mean(all_distances)
    hd95 = np.percentile(all_distances, 95)
    asd_95 = np.percentile(all_distances, 95)

    return {
        'ASD': asd,
        'HD95': hd95,
        'ASD_95': asd_95
    }


def cal_hd95(output, target, spacing=(1.0, 1.0, 1.0)):
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
    if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
        return 0.0, 0.0
    elif np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return 100.0, 100.0

    points_gt = compute_surface_points(mask_gt, spacing)
    points_pred = compute_surface_points(mask_pred, spacing)

    if len(points_gt) == 0 and len(points_pred) == 0:
        return 0.0, 0.0
    elif len(points_gt) == 0 or len(points_pred) == 0:
        return 100.0, 100.0

    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)

    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])

    hd95 = np.percentile(all_distances, 95)

    asd = np.mean(all_distances)

    return hd95, asd


def softmax_output_miou_class4(output, target):
    eps = 1e-8

    def compute_iou(o, t):
        intersect = torch.sum(o * t, dim=(1, 2, 3))
        union = torch.sum(o, dim=(1, 2, 3)) + torch.sum(t, dim=(1, 2, 3)) - intersect
        return (intersect + eps) / (union + eps)

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

    miou = (iou_ncr + iou_edema + iou_enhancing) / 3.0

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
        tp = torch.sum(o * t, dim=(1, 2, 3))
        fn = torch.sum((1 - o) * t, dim=(1, 2, 3))
        return (tp + eps) / (tp + fn + eps)

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
        dataname='BRATS2020',
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

    if dataname in ['dataset/Brast2024_MET', 'dataset/BraTS2020_TrainingData', 'dataset/Brast2023_PED_trainingdata']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro', 'ncr_net', 'edema', 'enhancing'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

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
                    weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training = False

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size]
                    pred_part = model(x_input, None, None, mask)
                    pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['/data2/xy/dataset/Brast2024_MET', '/data2/xy/M2FTrans/Dataset/BraTS2020_TrainingData',
                        '/data2/xy/dataset/Brast2023_PED/Brast2023_PED_trainingdata']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)
            vals_evaluation.update(scores_evaluation[k])

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    model.train()
    return vals_evaluation.avg


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