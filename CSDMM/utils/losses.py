from typing import List, Tuple, cast
# from functools import reduce
from operator import add
from functools import reduce
import numpy as np
import torch
from torch import einsum
from torch import Tensor
from torch import nn
import pandas as pd
import torch.nn.functional as F
from operator import mul

from einops import rearrange
from typing import Optional, Sequence
import pdb

import torch.nn.modules.padding


class Weighted_self_entropy_loss():

    # def self_entropy_loss(self, probs):
    #     log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
    #     mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
    #     mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
    #     loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
    #     loss /= mask.sum() + 1e-10
    #     return loss

    def max_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).sum(1)

    def tent_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x * torch.log(x + 1e-10)).sum(1).mean()

    def forward(self, pred: Tensor) -> Tensor:
        loss_entropy = self.tent_entropy(pred)
        # #### self-entropy loss
        # if self.act == 'softmax':
        #     pred = pred.softmax(1)
        #     loss_entropy = self.tent_entropy(pred)
        # else:
        #     pred = pred.sigmoid()
        #     ############ splitting type1
        #     # print('******splitting type1********')
        #     # union = pred.max(1)[0]
        #     # label_2 = (pred[:, 1] > 0.5).int()
        #     # pred_2 = label_2 * union
        #     # pred_1 = union - pred_2
        #     # pred = torch.stack([pred_1, pred_2], dim=1)
        #     ############# splitting type2
        #     # print('******splitting type2********')
        #     # pred_1 = pred[:, 0] - pred[:, 1]
        #     # pred_1 = torch.where(pred_1 >= 0.5, pred_1, torch.zeros_like(pred_1).to(pred_1.device))
        #     # pred_2 = pred[:, 1]
        #     # pred = torch.stack([pred_1, pred_2], dim=1)
        #     # label_1 = (pred[:, 0] > 0.5).int() - (pred[:, 1] > 0.5).int()
        #     # pred_1 = label_1 * pred[:, 0]
        #     # pred_2 = pred[:, 1]
        #     # # # pred_1 = pred[:, 0] - pred[:, 1]
        #     # # # pred_2 = pred[:, 1]
        #     # # # pred_3 = torch.ones_like(pred[:, 0]) - torch.max(pred, dim=1)[0]
        #
        #     loss_entropy = self.self_entropy_loss(pred)
        #     # loss_entropy = self.max_entropy(pred).mean()
        return loss_entropy

class KL_class_ratio_entropy_loss():
    def __init__(self):
        self.alpha = 0.1
        self.class_ratio_prior = torch.tensor([1/4, 1/4, 1/4, 1/4], dtype=torch.float32)
    def self_entropy_loss(self, probs):
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        # # 根据预测的类别数自适应的调整类别熵的权重
        # class_volume = probs.sum(dim=[0, 2, 3, 4])  # 1
        # inv_vol = 1.0 / torch.sqrt(class_volume + 1e-10)  # 2平滑变体
        # weight = inv_vol / inv_vol.mean()  # 3
        # weight = weight.detach()
        # mask_weighted = torch.einsum("bchwd,c->bchwd", [mask, weight.to(mask.device)])  # 4
        # 原始的
        mask_weighted = torch.einsum("bchwd,c->bchwd", [mask, Tensor([1,5,1,15]).to(mask.device)])
        loss = - torch.einsum("bchwd,bchwd->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss

    def forward(self, pred: Tensor) -> Tensor:
        b, k, h, w, d = pred.shape
        #### self-entropy loss

        loss_entropy = self.self_entropy_loss(pred)


        # pred = pred.sigmoid()
        batch_ratio = torch.zeros([b, 4]).to(pred.device).to(torch.float32)
        for i, c in enumerate([0,1,2,3]):
            class_sum = pred[:, c].sum([-3, -2, -1])
            batch_ratio[:, i] = class_sum / (h*w*d)
            # print(i, class_sum/(h*w))

        batch_ratio = batch_ratio.mean(0)
        self.class_ratio_prior = self.class_ratio_prior.to(pred.device)
        # # moving average的先验更新
        # self.class_ratio_prior = (1 - self.alpha) * self.class_ratio_prior + self.alpha * batch_ratio.detach()

        loss_shape_order_0 = torch.kl_div(batch_ratio.log(), self.class_ratio_prior).mean()

        # batch_ratio = batch_ratio
        # loss_shape_order_0 = self.prop_prior_loss(est_prop=batch_ratio, gt_prop=self.class_ratio_prior.to(pred.device))
        return loss_entropy


class RN_w_CR_loss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.k = 4
        self.d = 4
        self.M_k = nn.MaxPool3d(self.k, stride=self.k)
        self.M_d = nn.MaxPool3d(2 * self.d + 1, stride=2 * self.d + 1)

    def RN_loss(self, pred):
        """
        Regional Nuclear-Norm Loss: https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24
        https://github.com/cuishuhao/BNM
        The input should be [N, C]
        """
        pred = self.M_k(pred)
        C = pred.shape[0]
        pred = pred.transpose(1, 0).reshape(C, -1).T
        L_BNM = -torch.norm(pred, 'nuc')
        return L_BNM

    def CR_loss(self, pred):
        """
        Contour Regularization Loss.: https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24
        """
        max = self.M_d(pred)
        min = self.M_d(-pred)
        C_d = max + min
        C_d = torch.norm(C_d, dim=1, p=2)
        L_CR = C_d.mean()
        return L_CR

    def forward(self, pred: Tensor) -> Tensor:
        rn_l = self.RN_loss(pred)
        cr_l = self.CR_loss(pred)

        return 0.001 * rn_l + cr_l


class MS_loss():
    def __init__(self):
        pass

    def forward(self, pred: Tensor, image: Tensor) -> Tensor:
        B, C, H, W, D = pred.shape
        if image.dim() == 4:
            pred_centers = torch.sum((pred * image.view(B,1,H,W,D).expand(B,C,H,W,D)), dim=[2, 3, 4]) / torch.sum((pred), dim=[2, 3, 4])
            pred_centers = pred_centers.view(B, C, 1, 1, 1).expand(B, C, H, W, D).to(pred.device)
            mumford_loss = (image - pred_centers) ** 2 * pred
            mumford_loss = torch.sum(mumford_loss)
        elif image.dim() == 5:
            mumford_loss = torch.zeros(1).cuda().float()
            for sep in range(image.shape[1]):
                image_sep = image[:, sep]
                pred_centers = torch.sum((pred * image_sep.view(B, 1, H, W, D).expand(B, C, H, W, D)),
                                         dim=[2, 3, 4]) / torch.sum((pred), dim=[2, 3, 4])
                pred_centers = pred_centers.view(B, C, 1, 1, 1).expand(B, C, H, W, D).to(pred.device)
                loss = (image_sep - pred_centers) ** 2 * pred
                loss = torch.sum(loss)
                mumford_loss = mumford_loss + loss
            mumford_loss = mumford_loss / 4

        return mumford_loss

class NCCLoss3D(nn.Module):
    """Normalized Cross-Correlation loss for 3D images."""

    def __init__(self, eps=1e-10):
        super(NCCLoss3D, self).__init__()
        self.eps = eps

    def forward(self, img1, img2):
        """
        img1, img2: [B, C, D, H, W] or [B, 1, D, H, W]
        """
        # mean over spatial dims
        mean1 = torch.mean(img1, dim=[-1, -2, -3], keepdim=True)
        mean2 = torch.mean(img2, dim=[-1, -2, -3], keepdim=True)

        v1 = img1 - mean1
        v2 = img2 - mean2

        numerator = torch.sum(v1 * v2, dim=[-1, -2, -3])
        denominator = torch.sqrt(
            torch.sum(v1 ** 2, dim=[-1, -2, -3]) *
            torch.sum(v2 ** 2, dim=[-1, -2, -3]) + self.eps
        )

        return (1 - numerator / denominator).mean()

def l2_reg_ortho(model):
    """
    Compute orthogonality regularization penalty:
    || Wᵀ W - I || (largest singular value)
    """
    device = next(model.parameters()).device
    l2_reg = None

    for W in model.parameters():
        if W.ndimension() < 2:
            continue

        # W: [out_channels, in_channels, ...] → reshape to 2D weight matrix
        cols = W[0].numel()
        w1 = W.view(-1, cols)   # Shape: [rows, cols]
        wt = w1.t()             # [cols, rows]

        m = torch.matmul(wt, w1)    # Wᵀ W
        ident = torch.eye(cols, device=device)  # Identity matrix

        w_tmp = m - ident

        # Power iteration to approximate largest singular value
        height = w_tmp.size(0)
        u = torch.randn(height, device=device)
        u = F.normalize(u, dim=0, eps=1e-12)
        v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)

        sigma = torch.dot(u, torch.matmul(w_tmp, v))

        if l2_reg is None:
            l2_reg = sigma ** 2
        else:
            l2_reg = l2_reg + sigma ** 2

    return l2_reg


class EntKLPropWMoment3D():
    def __init__(self,num_cls):
        self.alpha = 0.1
        self.num_cls = num_cls

        if self.num_cls==4:
            self.class_ratio_prior = torch.tensor(
                [1/4, 1/4, 1/4, 1/4],
                dtype=torch.float32
            )
        if self.num_cls==2:
            self.class_ratio_prior = torch.tensor(
                [1 / 2, 1 / 2],
                dtype=torch.float32
            )

        # moment先验（动态）
        self.mom_est = None

        self.margin = 0.1
        self.lambda_moment = 0.1


    def self_entropy_loss(self, probs):
        log_p = (probs + 1e-10).log()
        mask = probs.float()

        if self.num_cls == 4:
            mask_weighted = torch.einsum(
                "bchwd,c->bchwd",
                [mask, torch.tensor([1,5,1,15]).to(mask.device)]
            )
        elif self.num_cls == 2:
            mask_weighted = torch.einsum(
                "bchwd,c->bchwd",
                [mask, torch.tensor([1, 1]).to(mask.device)]
            )

        loss = - torch.einsum("bchwd,bchwd->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss


    def kl_loss(self, probs):
        b, c, h, w, d = probs.shape

        batch_ratio = probs.sum(dim=[2,3,4]) / (h*w*d)
        batch_ratio = batch_ratio.mean(0)

        self.class_ratio_prior = self.class_ratio_prior.to(probs.device)

        loss_kl = torch.kl_div(
            batch_ratio.log(),
            self.class_ratio_prior
        ).mean()

        return loss_kl


    def moment_3d(self, probs):
        """
        计算简单3D moment:
        size + centroid (x,y,z)
        """
        b, c, h, w, d = probs.shape

        grid_x = torch.arange(w, device=probs.device)
        grid_y = torch.arange(h, device=probs.device)
        grid_z = torch.arange(d, device=probs.device)

        yy, xx, zz = torch.meshgrid(grid_y, grid_x, grid_z, indexing="ij")

        xx = xx.float()
        yy = yy.float()
        zz = zz.float()

        M000 = probs.sum(dim=[2, 3, 4]) / (h * w * d) + 1e-5

        M100 = (probs * xx).sum(dim=[2, 3, 4])
        M010 = (probs * yy).sum(dim=[2, 3, 4])
        M001 = (probs * zz).sum(dim=[2, 3, 4])

        cx = (M100 / (M000 * (h * w * d))) / w
        cy = (M010 / (M000 * (h * w * d))) / h
        cz = (M001 / (M000 * (h * w * d))) / d

        # 拼接 moment
        moment = torch.stack([M000, cx, cy, cz], dim=-1)  # (b,c,4)

        return moment

    def moment_loss(self, probs):
        probs_moment = self.moment_3d(probs)  # (b,c,4)

        # ===== EMA 更新 =====
        batch_moment = probs_moment.mean(dim=0)  # (c,4)

        if self.mom_est is None:
            self.mom_est = batch_moment.detach()
        else:
            self.mom_est = (
                0.9 * self.mom_est +
                0.1 * batch_moment.detach()
            )

        # ===== 构造约束 =====
        b, c, t = probs_moment.shape

        est = self.mom_est.unsqueeze(0).expand(b, -1, -1)

        upper = est * (1 + self.margin)
        lower = est * (1 - self.margin)

        upper_penalty = F.relu(upper - probs_moment) ** 2
        lower_penalty = F.relu(probs_moment - lower) ** 2

        loss = upper_penalty + lower_penalty

        return loss.mean()

    def forward(self, probs):
        loss_entropy = self.self_entropy_loss(probs)
        loss_kl = self.kl_loss(probs)
        loss_moment = self.moment_loss(probs)

        total_loss = loss_entropy + loss_kl + self.lambda_moment * loss_moment

        return total_loss