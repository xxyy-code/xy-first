#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict
import csv
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.losses import Weighted_self_entropy_loss, KL_class_ratio_entropy_loss, RN_w_CR_loss, MS_loss
import mmformer_moe2
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='CW-TTA (class-wise distribution alignment)')

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--dataname', required=True, type=str,
                    help='Target domain root containing vol/ and seg/')
parser.add_argument('--train_file', required=True, type=str,
                    help='Target train/adapt split list (.txt)')
parser.add_argument('--test_file', required=True, type=str,
                    help='Target test/eval split list (.txt)')
parser.add_argument('--source_root', required=True, type=str,
                    help='Source domain root used to estimate class priors')
parser.add_argument('--source_train_file', required=True, type=str,
                    help='Source train split list (.txt)')
parser.add_argument('--pretrain', required=True, type=str,
                    help='Path to pretrained CSD-MM checkpoint (.pth)')
parser.add_argument('--savepath', default='./output/ada_pth', type=str)
parser.add_argument('--csv_path', default='./output/csv/cw_tta.csv', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--num_cls', default=4, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--iter_per_epoch', default=99, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--crop_size', default=80, type=int, help='Adaptation random crop size')
parser.add_argument('--rotate_angle', default=10, type=float, help='Random rotation degrees')
parser.add_argument('--intensity_shift', default=0.1, type=float, help='Random intensity shift/scale')
parser.add_argument('--align_tumor_weight', default=10.0, type=float,
                    help='Extra weight on tumor-related moment alignment terms')
parser.add_argument('--mass_delta_ratio', default=0.3, type=float,
                    help='Relative mass/entropy change threshold for bad-class detection')
parser.add_argument('--eval_every', default=2, type=int, help='Run eval every N epochs')
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
cs = int(args.crop_size)
args.train_transforms = (
    f'Compose([RandCrop3D(({cs},{cs},{cs})), RandomRotion({args.rotate_angle}), '
    f'RandomIntensityChange(({args.intensity_shift},{args.intensity_shift})), '
    f'RandomFlip(0), NumpyType((np.float32, np.int64)),])'
)
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.csv_path)) or '.', exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['flairt1cet1t2']
print (masks_torch.int())
def update_ema_model(teacher, student, alpha=0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data
def source_class_channel_features(model, loader, i, feat_C=16, skip_background=False):
    class_de_x1 = defaultdict(list)

    data_iter = iter(loader)
    with torch.no_grad():
        for _ in range(len(loader)):
            data = next(data_iter)
            x, target,_, mask = data[:4]

            x = x.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target[i].cuda(non_blocking=True)

            model.module.is_training = True
            _, _, _, _, de_x = model(x,_,_, mask)
            de_x1 = de_x[i]
            target_idx = torch.argmax(target, dim=1)

            target_flat = target_idx.view(-1).cpu().numpy()
            de_x1_flat = (
                de_x1.permute(0, 2, 3, 4, 1)
                .reshape(-1, feat_C)
                .cpu()
                .numpy()
            )

            for cls in np.unique(target_flat):

                if cls == 0 and skip_background:
                    continue
                cls_mask = target_flat == cls
                if cls_mask.sum() < 10:
                    continue
                class_de_x1[cls].append(de_x1_flat[cls_mask])

    # merge
    class_voxels = {}
    for cls, batch_list in class_de_x1.items():
        if not batch_list:
            continue
        all_for_cls = np.concatenate(batch_list, axis=0)
        class_voxels[cls] = [all_for_cls[:, ch] for ch in range(feat_C)]

    # 计算每个类别每个通道的均值和方差
    class_channel_means = {}
    class_channel_vars = {}  # 新增：方差

    for cls, channels_list in class_voxels.items():
        class_channel_means[cls] = []
        class_channel_vars[cls] = []

        for ch in range(feat_C):
            data = channels_list[ch]
            if len(data) == 0:
                mean_val = np.nan
                var_val = np.nan
            else:
                mean_val = np.mean(data)
                var_val = np.var(data)  # 默认使用总体方差 (ddof=0)
                # 如果想要样本方差，可以用 np.var(data, ddof=1)

            class_channel_means[cls].append(mean_val)
            class_channel_vars[cls].append(var_val)

    return class_channel_means, class_channel_vars

# def target_class_channel_features(de_x1, pes_label, cls_list,feat_C=16, skip_background=True):
#     class_de_x1 = defaultdict(list)
#     target_idx = torch.argmax(pes_label, dim=1)
#
#     target_flat = target_idx.view(-1).numpy()
#     de_x1_flat = (
#         de_x1.permute(0, 2, 3, 4, 1)
#         .reshape(-1, feat_C)
#         .cpu()
#         .numpy()
#     )
#
#     for cls in cls_list:
#
#         if cls == 0 and skip_background:
#             continue
#         cls_mask = target_flat == cls
#         if cls_mask.sum() < 10:
#             class_de_x1[cls].append('no')
#         class_de_x1[cls].append(de_x1_flat[cls_mask])
#
#     # merge
#     class_voxels = {}
#     for cls, batch_list in class_de_x1.items(): # 类，特征
#         if batch_list != 'no':
#             continue
#         all_for_cls = np.concatenate(batch_list, axis=0)
#         class_voxels[cls] = [all_for_cls[:, ch] for ch in range(feat_C)]
#
#     # 计算每个类别每个通道的均值和方差
#     class_channel_means = {}
#     class_channel_vars = {}  # 新增：方差
#
#     for cls, channels_list in class_voxels.items():
#         class_channel_means[cls] = []
#         class_channel_vars[cls] = []
#
#         for ch in range(feat_C):
#             data = channels_list[ch]
#             if len(data) == 0:
#                 mean_val = np.nan
#                 var_val = np.nan
#             else:
#                 mean_val = np.mean(data)
#                 var_val = np.var(data)  # 默认使用总体方差 (ddof=0)
#                 # 如果想要样本方差，可以用 np.var(data, ddof=1)
#
#             class_channel_means[cls].append(mean_val)
#             class_channel_vars[cls].append(var_val)
#
#     return class_de_x1, class_channel_means, class_channel_vars

def target_class_channel_features(
    de_x1, pes_label, cls_list,
    feat_C=16, skip_background=True
):
    device = de_x1.device
    class_mu = {}
    class_var = {}

    target_idx = torch.argmax(pes_label, dim=1)  # [B,H,W,D]

    for cls in cls_list:
        if cls == 0 and skip_background:
            class_mu[cls] = 'no'
            class_var[cls] = 'no'
        else:
            cls_mask = (target_idx == cls)  # [B,H,W,D]

            if cls_mask.sum() < 10:
                class_mu[cls] = 'no'
                class_var[cls] = 'no'
            else:
                # [N_voxels, C]
                feats = de_x1.permute(0,2,3,4,1)[cls_mask]

                mu = feats.mean(dim=0)                 # [feat_C]
                var = feats.var(dim=0, unbiased=False) # [feat_C]

                class_mu[cls] = mu
                class_var[cls] = var

    return class_mu, class_var
def moment_alignment_loss(mu_s, var_s, mu_t, var_t):
    """
    mu_s, mu_t: [C, F]
    std_s, std_t: [C, F]
    """
    mean_loss = 0.5 * torch.mean((mu_s - mu_t) ** 2)
    std_loss  = 0.5 * torch.mean((var_s - var_t) ** 2)

    return mean_loss + std_loss


def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    num_cls = args.num_cls
    model = mmformer_moe2.Model(num_cls=num_cls)
    source_model = mmformer_moe2.Model(num_cls=num_cls)
    pretrained_path = args.pretrain
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f'pretrain checkpoint not found: {pretrained_path}')
    pretrained_params = torch.load(pretrained_path, map_location='cpu')
    state_dict = pretrained_params['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    source_model.load_state_dict(new_state_dict)
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in source_model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'e1_c1' in name or 'e1_c2' in name or 'e1_c3' in name or 'e2_c1' in name or 'e2_c2' in name or 'e2_c3' in name or 'e3_c1' in name or 'e3_c2' in name or 'e3_c3' in name:
        # if 'encoder' in name:
            param.requires_grad = True
            print("Trainable:", name)

    model = torch.nn.DataParallel(model).cuda()
    source_model = torch.nn.DataParallel(source_model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    logging.info(str(args))
    source_data = Brats_loadall_nii(
        transforms=args.train_transforms,
        root=args.source_root,
        num_cls=num_cls,
        train_file=args.source_train_file,
    )
    train_set = Brats_loadall_nii(
        transforms=args.train_transforms,
        root=args.dataname,
        num_cls=num_cls,
        train_file=args.train_file,
    )
    test_set = Brats_loadall_test_nii(
        transforms=args.test_transforms,
        root=args.dataname,
        test_file=args.test_file,
    )
    source_loader = MultiEpochsDataLoader(
        dataset=source_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # 先提取源域的特征分布信息
    source_mu, source_var = source_class_channel_features(source_model, source_loader, 0,feat_C = 16, skip_background=False)
    source_bg_mu, source_bg_var = source_class_channel_features(source_model, source_loader, 1,feat_C = 16, skip_background=True)
    source_ncr_mu, source_ncr_var = source_class_channel_features(source_model, source_loader, 2,feat_C = 16, skip_background=True)
    source_edema_mu, source_edema_var = source_class_channel_features(source_model, source_loader, 3,feat_C = 16, skip_background=True)
    source_enhan_mu, source_enhan_var = source_class_channel_features(source_model, source_loader, 4,feat_C = 16, skip_background=True)
    print('source加载完成')

    #########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        # test_score = AverageMeter()
        # with torch.no_grad():
        #     logging.info('###########test set wi post process###########')
        #     for i, mask in enumerate(masks[::-1]):
        #         logging.info('{}'.format(mask_name[::-1][i]))
        #         dice_score = test_softmax(
        #                         test_loader,
        #                         model,
        #                         dataname = args.dataname,
        #                         feature_mask = mask,
        #                         mask_name = mask_name[::-1][i])
        #         test_score.update(dice_score)
        #     logging.info('Avg scores: {}'.format(test_score.avg))
        #     exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch

    train_iter = iter(train_loader)
    bad_classes = []
    weight = [1,1,1,1]
    for epoch in range(args.num_epochs):

        step_lr = lr_schedule(optimizer, epoch)
        b = time.time()
        for i in range(iter_per_epoch):
            print(f'开始训练{i}/{epoch}/{args.num_epochs}')
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, _,_, mask = data[:4]
            x = x.cuda(non_blocking=True)

            mask = mask.cuda(non_blocking=True)
            model.module.is_training = True
            source_model.module.is_training = True

            fuse_pred, sep_preds, prm_preds, bi_outputs, de_xs = model(x,_,_, mask)

            pesudo_label = torch.argmax(fuse_pred, dim=1)
            mask_non_enh = torch.ones_like(pesudo_label, dtype=torch.bool)

            for i in bad_classes:
                mask_non_enh &= (pesudo_label != i)

            fuse_mu, fuse_var = target_class_channel_features(de_xs[0] * mask_non_enh, fuse_pred, [0, 1, 2, 3], skip_background=False)
            bg_mu, bg_var = target_class_channel_features(de_xs[1], bi_outputs[0], [0, 1], skip_background=True)
            ncr_mu, ncr_var = target_class_channel_features(de_xs[2], bi_outputs[1], [0, 1], skip_background=True)
            edema_mu, edema_var = target_class_channel_features(de_xs[3], bi_outputs[2], [0, 1], skip_background=True)
            enhan_mu, enhan_var = target_class_channel_features(de_xs[4], bi_outputs[3], [0, 1], skip_background=True)

            loss_pred = torch.zeros(1).cuda().float()
            # 四分类对齐
            for j in [0, 1, 2, 3]:
                if fuse_mu[j] != 'no':
                    loss_fuse = moment_alignment_loss(torch.tensor(source_mu[j]).cuda(),
                                                      torch.tensor(source_var[j]).cuda(), fuse_mu[j], fuse_var[j])
                else:
                    loss_fuse = 0
                if j == 1 or j == 3:
                    loss_pred += args.align_tumor_weight * loss_fuse
            loss_bi = torch.zeros(1).cuda().float()
            for j in [0, 1]:
                if bg_mu[j] != 'no':
                    loss_bg = moment_alignment_loss(torch.tensor(source_bg_mu[j]).cuda(),
                                                    torch.tensor(source_bg_var[j]).cuda(), bg_mu[j], bg_var[j])
                else:
                    loss_bg = 0
                if ncr_mu[j] != 'no':
                    if j == 0:
                        wei = weight[0]
                    else:
                        wei = weight[1]
                    loss_ncr = wei * moment_alignment_loss(torch.tensor(source_ncr_mu[j]).cuda(),
                                                     torch.tensor(source_ncr_var[j]).cuda(), ncr_mu[j], ncr_var[j])
                else:
                    loss_ncr = 0
                if edema_mu[j] != 'no':
                    if j == 0:
                        wei = weight[0]
                    else:
                        wei = weight[2]
                    loss_edema = wei * moment_alignment_loss(torch.tensor(source_edema_mu[j]).cuda(),
                                                       torch.tensor(source_edema_var[j]).cuda(), edema_mu[j],
                                                       edema_var[j])
                else:
                    loss_edema = 0
                if enhan_mu[j] != 'no':
                    if j == 0:
                        wei = weight[0]
                    else:
                        wei = weight[2]
                    loss_enhan = wei * moment_alignment_loss(torch.tensor(source_enhan_mu[j]).cuda(),
                                                       torch.tensor(source_enhan_var[j]).cuda(), enhan_mu[j],
                                                       enhan_var[j])
                else:
                    loss_enhan = 0
                loss_bi += loss_bg + loss_edema + args.align_tumor_weight * (loss_enhan + loss_ncr)
            loss = loss_pred / 4 + loss_bi / 8

            optimizer.zero_grad()
            loss.backward()

            # # --- 每个参数梯度单独归一化 ---
            # for p in model.parameters():
            #     if p.grad is not None:
            #         grad_norm = p.grad.data.norm(2)  # L2 norm of this parameter's grad
            #         if grad_norm > 0:
            #             p.grad.data /= grad_norm

            optimizer.step()

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            # msg += 'fuseloss:{:.4f},'.format(loss_pred.item())
            # msg += 'biloss:{:.4f},'.format(loss_bi.item())

            logging.info(msg)

        student_mass = torch.zeros(num_cls).cuda()
        teacher_mass = torch.zeros(num_cls).cuda()

        student_entropy = torch.zeros(num_cls).cuda()
        teacher_entropy = torch.zeros(num_cls).cuda()

        student_count = torch.zeros(num_cls).cuda()
        teacher_count = torch.zeros(num_cls).cuda()
        if (epoch + 1) % args.eval_every == 0:
            eps = 1e-8
            with torch.no_grad():
                for i, data in enumerate(train_loader):
                    x, _,_, mask = data[:4]
                    x = x.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                    fuse_pred_s, _, _, _, _ = model(x,_,_, mask)
                    fuse_pred_t, _, _, _, _ = source_model(x,_,_, mask)
                    # [B, C, H, W, D] -> [N, C]
                    ps = fuse_pred_s.permute(0, 2, 3, 4, 1).reshape(-1, num_cls)
                    pt = fuse_pred_t.permute(0, 2, 3, 4, 1).reshape(-1, num_cls)
                    # 熵
                    Hs = -(ps * torch.log(ps + eps)).sum(dim=1)
                    Ht = -(pt * torch.log(pt + eps)).sum(dim=1)

                    # teacher 的 hard label 作为 anchor
                    y_t = torch.argmax(pt, dim=1)

                    for c in range(num_cls):
                        if c == 0:
                            continue
                        mask_c = (y_t == c)

                        if mask_c.sum() == 0:
                            continue

                        # 体素比例 / soft mass
                        student_mass[c] += ps[mask_c, c].sum()
                        teacher_mass[c] += pt[mask_c, c].sum()

                        # 熵
                        student_entropy[c] += Hs[mask_c].sum()
                        teacher_entropy[c] += Ht[mask_c].sum()

                        student_count[c] += mask_c.sum()
                        teacher_count[c] += mask_c.sum()
                mean_student_mass = student_mass / (student_count + eps)
                mean_teacher_mass = teacher_mass / (teacher_count + eps)

                mean_student_entropy = student_entropy / (student_count + eps)
                mean_teacher_entropy = teacher_entropy / (teacher_count + eps)

                delta_mass = torch.abs(mean_student_mass - mean_teacher_mass)
                delta_entropy = torch.abs(mean_student_entropy - mean_teacher_entropy)
                print('epoch', epoch)
                print('delta_mass', delta_mass)
                print('delta_entropy', delta_entropy)
            bad_classes = []

            for c in range(num_cls):
                if c == 0:
                    continue
                thr = args.mass_delta_ratio
                # if delta_mass[c] > thr * mean_teacher_mass[c] or delta_entropy[c] > thr * mean_teacher_entropy[c]:
                #     bad_classes.append(c)
                if delta_mass[c] > thr * mean_teacher_mass[c]:
                    bad_classes.append(c)
                if delta_entropy[c] > thr * mean_teacher_entropy[c]:
                    ratio = (
                            (delta_entropy[c] - thr * mean_teacher_entropy[c])
                            / (thr * mean_teacher_entropy[c] + eps)
                    )

                    weight[c] = torch.exp(-1 * ratio)
            if len(bad_classes) > 0:
                print(f"Reject epoch, unstable classes: {bad_classes}")

                # 1️⃣ 回滚整个模型
                model.load_state_dict(source_model.state_dict())
            else:
                print("Accept epoch update")

                # 1️⃣ 更新 teacher
                source_model.load_state_dict(model.state_dict())
                # update_ema_model(source_model, model, alpha=0.99)

        # print(f'结束第{epoch}个训练')
        # logging.info('train time per epoch: {}'.format(time.time() - b))

                ##########model save
                file_name = os.path.join(ckpts, 'model_moegw_ada_dis_align_all_de_{}.pth'.format(epoch+1))
                torch.save({
                    'epoch': epoch,
                    'state_dict': source_model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)
                ##########Evaluate the epoch model
                test_score = AverageMeter()
                hd95_score = AverageMeter()
                checkpoint_path = file_name
                checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 如果用GPU，改成cuda()
                source_model.load_state_dict(checkpoint['state_dict'])
                source_model.eval()
                class_names = ['whole', 'core', 'enhancing', 'enhancing_postpro', 'ncr_net', 'edema', 'enhancing']
                with torch.no_grad():
                    logging.info('###########test set wi/wo postprocess###########')
                    for i, mask in enumerate(masks[::-1]):
                        logging.info('{}'.format(mask_name[::-1][i]))
                        dice_score, hd95 = test_softmax(
                            test_loader,
                            source_model,
                            dataname=args.dataname,
                            feature_mask=mask)
                        test_score.update(dice_score)
                        hd95_score.update(hd95)

                # 保存平均分
                # 是否需要写表头

                save_path = args.csv_path
                write_header = not os.path.exists(save_path)

                with open(save_path, "a", newline="") as f:  # 用 "a" 追加，而不是 "w"
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['epoch'] + class_names)
                    writer.writerow([ + 1] + test_score.avg.tolist())
                    writer.writerow([ + 1] + hd95_score.avg.tolist())
                    # if (test_score.avg > best_dice).all():
                    #     best_dice = test_score.avg.copy()
                    #     file_name = os.path.join(ckpts, 'model_PASS_best.pth')
                    #     torch.save({
                    #         'epoch': epoch,
                    #         'state_dict': model.state_dict(),
                    #         'optim_dict': optimizer.state_dict(),
                    #     },
                    #         file_name)

    # msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    # logging.info(msg)


if __name__ == '__main__':
    main()