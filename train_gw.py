#coding=utf-8
#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mmformer_moe2
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='CSD-MM pretraining')

parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--dataname', required=True, type=str,
                    help='Source data root containing vol/ and seg/')
parser.add_argument('--train_file', required=True, type=str,
                    help='Train split list (.txt), absolute path or relative to dataname')
parser.add_argument('--test_file', required=True, type=str,
                    help='ID test split list (.txt), absolute path or relative to dataname')
parser.add_argument('--ood_root', default='', type=str,
                    help='Optional OOD data root for periodic eval (e.g. MET)')
parser.add_argument('--ood_test_file', default='', type=str,
                    help='Optional OOD split list')
parser.add_argument('--savepath', default='./output/pth', type=str, help='Checkpoint directory')
parser.add_argument('--csv_path', default='./output/csv/moegw.csv', type=str, help='Eval log csv')
parser.add_argument('--resume', default='', type=str, help='Resume checkpoint (optional)')
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--eval_ckpt', default='', type=str,
                    help='Optional ckpt for final eval; empty = skip final hard-coded eval')
parser.add_argument('--num_cls', default=4, type=int, help='Number of classes (BraTS=4)')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=110, type=int)
parser.add_argument('--region_fusion_start_epoch', default=34, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--num_workers', default=8, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.csv_path)) or '.', exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
# masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]]
masks = [[True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
# mask_name = ['t2', 't1c', 't1', 'flair',
#             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
#             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
#             'flairt1cet1t2']
mask_name = ['flairt1cet1t2']
print (masks_torch.int())

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
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.dataname, num_cls=num_cls,
                                  train_file=args.train_file)
    test_set1 = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.dataname, test_file=args.test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader1 = MultiEpochsDataLoader(
        dataset=test_set1,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    test_loader2 = None
    if args.ood_root and args.ood_test_file:
        test_set2 = Brats_loadall_test_nii(
            transforms=args.test_transforms,
            root=args.ood_root,
            test_file=args.ood_test_file,
        )
        test_loader2 = MultiEpochsDataLoader(
            dataset=test_set2,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    # ########Evaluate
    # if args.resume is not None:
    #     if not os.path.isfile(args.resume):
    #         raise FileNotFoundError(f"检查点文件不存在: {args.resume}")
    #
    #     checkpoint = torch.load(args.resume)
    #
    #     # 先创建优化器（结构必须和保存时完全一样）
    #     optimizer = torch.optim.Adam(
    #         [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}],
    #         betas=(0.9, 0.999), eps=1e-08, amsgrad=True
    #     )
    #
    #     # 恢复优化器状态
    #     if 'optim_dict' in checkpoint:
    #         # 加载完 optimizer 后立即执行
    #         old_lr = optimizer.param_groups[0]['lr']
    #         new_lr = old_lr * 0.05  # 先试 1/20，极端情况可试 0.01 或 0.001
    #         for group in optimizer.param_groups:
    #             group['lr'] = new_lr
    #         print(f"恢复后强制降低 lr: {old_lr:.2e} → {new_lr:.2e}")
    #     else:
    #         print("警告：检查点中没有优化器状态，将从初始状态继续")
    #
    #     # 恢复模型
    #     model.load_state_dict(checkpoint['state_dict'])
    #
    #     start_epoch = checkpoint['epoch'] + 1
    #     logging.info(f"从 epoch {checkpoint['epoch']} 恢复，继续训练 (start_epoch={start_epoch})")
    #
    # else:
    #     start_epoch = 0
    #     print("从头开始训练")

    #########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    # iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    print(train_iter)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            print(f'开始训练{i}/{epoch}/{args.num_epochs}')
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, boundarys, mask = data[:4]
            x = x.cuda(non_blocking=True)
            target[0] = target[0].cuda(non_blocking=True)
            target[1] = target[1].cuda(non_blocking=True)
            target[2] = target[2].cuda(non_blocking=True)
            target[3] = target[3].cuda(non_blocking=True)
            target[4] = target[4].cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            bis = (target[2], target[3], target[4])

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, bi_masks, rou_logits, gw_loss = model(x,boundarys,bis , mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target[0], num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target[0], num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target[0], num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target[0], num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target[0], num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target[0], num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            bi_cross_loss = torch.zeros(1).cuda().float()
            bi_dice_loss = torch.zeros(1).cuda().float()
            j=1
            for bi_pred in bi_masks:

                bi_cross_loss += criterions.softmax_weighted_loss(bi_pred, target[j], num_cls=num_cls)
                bi_dice_loss += criterions.dice_loss(bi_pred, target[j], num_cls=num_cls)
                j+=1
            bi_loss = bi_cross_loss + bi_dice_loss

            rou_var_loss = torch.zeros(1).cuda().float()
            rou_entro_loss = torch.zeros(1).cuda().float()
            for rou_pred in rou_logits:
                rou_var_loss += rou_pred.var(dim=-1).mean()
                rou_softmax = F.softmax(rou_pred, dim=-1)
                rou_entro_loss += -(rou_softmax*torch.log(rou_softmax + 1e-9)).sum(dim=-1).mean()
            rou_loss = 0.01 * rou_var_loss + 0.005 * rou_entro_loss
            if isinstance(gw_loss, torch.Tensor) and gw_loss.dim() > 0:
                gw_loss = gw_loss.mean()
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + bi_loss + rou_loss + gw_loss * 0.0
            else:
                loss = fuse_loss + sep_loss + prm_loss + bi_loss + rou_loss + gw_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('bi_cross_loss', bi_cross_loss.item(), global_step=step)
            writer.add_scalar('bi_dice_loss', bi_dice_loss.item(), global_step=step)
            writer.add_scalar('rou_var_loss', rou_var_loss.item(), global_step=step)
            writer.add_scalar('rou_entrp_loss', rou_entro_loss.item(), global_step=step)
            writer.add_scalar('gw_loss', gw_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'bicross:{:.4f}, bidice:{:.4f},'.format(bi_cross_loss.item(), bi_dice_loss.item())
            msg += 'rouvar:{:.4f}, rouentro:{:.4f},'.format(rou_var_loss.item(), rou_entro_loss.item())
            msg += 'gwloss:{:.4f},'.format(gw_loss.item())
            logging.info(msg)
        print(f'结束第{epoch}个训练')
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_full_moe3.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        # if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
        #     file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optim_dict': optimizer.state_dict(),
        #         },
        #         file_name)
        if (epoch + 1) > 190 and (epoch + 1) % 100 == 0:
            file_name1 = os.path.join(ckpts, 'model_full_moe3_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name1)
            ##########Evaluate the last epoch model
            test_score1 = AverageMeter()
            test_score2 = AverageMeter()
            checkpoint_path = file_name1
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 如果用GPU，改成cuda()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            class_names = ['whole', 'core', 'enhancing', 'enhancing_postpro', 'ncr_net', 'edema', 'enhancing']
            with torch.no_grad():
                logging.info('###########test set wi/wo postprocess###########')
                for i, mask in enumerate(masks[::-1]):
                    logging.info('{}'.format(mask_name[::-1][i]))
                    dice_score = test_softmax(
                        test_loader1,
                        model,
                        dataname=args.dataname,
                        feature_mask=mask,
                    )
                    test_score1.update(dice_score)
                if test_loader2 is not None:
                    for i, mask in enumerate(masks[::-1]):
                        logging.info('{}'.format(mask_name[::-1][i]))
                        dice_score = test_softmax(
                            test_loader2,
                            model,
                            dataname=args.ood_root or args.dataname,
                            feature_mask=mask,
                        )
                        test_score2.update(dice_score)

            save_path = args.csv_path
            write_header = not os.path.exists(save_path)
            import csv
            with open(save_path, "a", newline="") as f:  # 用 "a" 追加，而不是 "w"
                writer1 = csv.writer(f)
                if write_header:
                    writer1.writerow(['epoch'] + class_names)
                writer1.writerow([epoch + 1] + test_score1.avg.tolist())
                if test_loader2 is not None:
                    writer1.writerow([epoch + 1] + test_score2.avg.tolist())

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

    ##########Optional final evaluation
    if args.eval_ckpt:
        test_score = AverageMeter()
        hd95_score = AverageMeter()
        checkpoint = torch.load(args.eval_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.no_grad():
            logging.info('###########final eval###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score, hd95 = test_softmax(
                                test_loader1,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask)
                test_score.update(dice_score)
                hd95_score.update(hd95)

if __name__ == '__main__':
    main()
