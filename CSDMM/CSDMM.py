import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm

basic_dims = 16
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 3
num_modals = 4
patch_size = 5


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=in_dim, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4)), de_x1


class Decoder_ex(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_ex, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)

        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))

        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))

        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))

        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, de_x1


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


# 专家选fea
class ExpertChoiceTokenNoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(ExpertChoiceTokenNoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(n_embed * 4, num_experts)
        self.noise_linear = nn.Linear(n_embed * 4, num_experts)

    def forward(self, mh_output):
        # mh_output: [B, 4C, H, W, D]
        global_feat = mh_output.mean(dim=(2, 3, 4))  # [B,4C]
        logits = self.topkroute_linear(global_feat)  # [B, num_experts]
        noise_logits = self.noise_linear(global_feat)  # [B, num_experts]

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices, noisy_logits


class expert(nn.Module):
    def __init__(self, num_cls):
        super(expert, self).__init__()
        self.trans = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                 mlp_dim=mlp_dim)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims * 3, basic_dims * 16 * 2,
                                                kernel_size=1, padding=0)
        self.decoder = Decoder_ex(num_cls=num_cls)

    def forward(self, router, indices, features, pos, high_feas):
        # 输出1：筛选有用模态（路由非0）
        B, _, C, H, W, D = features.shape
        idx = indices.view(B, 2, 1, 1, 1, 1)
        idx = idx.expand(-1, -1, C, H, W, D)  # [B,2,512,H,W,D]
        selected_feats = torch.gather(features, dim=1, index=idx)  # [B,2,512,H,W,D]
        # 输出2：根据路由做权重和
        weights = router.view(B, 4, 1, 1, 1, 1)
        fused_feat2 = (weights * features).sum(dim=1)  # [B,512,H,W,D]
        # 对输出1做transformer
        idx_pos = indices.view(B, 2, 1, 1)
        _, _, P, C = pos.shape
        idx_pos = idx_pos.expand(-1, -1, P, C)  # [B,2,P,C]
        multimodal_pos = torch.gather(pos, dim=1, index=idx_pos)  # [B,2,P,C]
        multimodal_pos = multimodal_pos.view(B, 2 * P, C)
        multimodal_token = selected_feats.permute(0, 1, 3, 4, 5, 2).contiguous().view(B, -1,
                                                                                      transformer_basic_dims)  # [B,2,HWD,512]
        multimodal_token = self.trans(multimodal_token, multimodal_pos)  # [B,2,HWD,512]
        fused_feat1 = multimodal_token.view(B, 2 * transformer_basic_dims, H, W, D)
        fused_feat = torch.cat((fused_feat1, fused_feat2), dim=1)
        fused_feat = self.multimodal_decode_conv(fused_feat)

        feas_list = []
        for feas in high_feas:
            b, c, h, w, d = feas.shape
            feas = feas.view(b, -1, c // 4, h, w, d)
            b, _, c, h, w, d = feas.shape
            idx_fea = indices.view(b, 2, 1, 1, 1, 1)
            idx_fea = idx_fea.expand(-1, -1, c, h, w, d)
            feas_list.append(torch.gather(feas, dim=1, index=idx_fea).view(b, -1, h, w, d))

        output, de_x1 = self.decoder(feas_list[0], feas_list[1], feas_list[2], feas_list[3], fused_feat)

        return output, de_x1


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder(1)
        self.t1ce_encoder = Encoder(1)
        self.t1_encoder = Encoder(1)
        self.t2_encoder = Encoder(1)

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.fuse_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))

        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                             mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                            mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                          mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                          mlp_dim=mlp_dim)
        self.fuse_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                            mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                                  mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims * 5, basic_dims * 16 * num_modals,
                                                kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.router_edema = ExpertChoiceTokenNoisyTopkRouter(512, 4, 2)
        self.router_ncr = ExpertChoiceTokenNoisyTopkRouter(512, 4, 2)
        self.router_enhan = ExpertChoiceTokenNoisyTopkRouter(512, 4, 2)
        self.router_bg = ExpertChoiceTokenNoisyTopkRouter(512, 4, 2)

        self.expert_edema = expert(2)
        self.expert_ncr = expert(2)
        self.expert_enhan = expert(2)
        self.expert_bg = expert(2)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)  #

    def forward(self, x, boundaries, bis, mask):
        # extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                                   transformer_basic_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                                transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          transformer_basic_dims)

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        # [B,512,5,5,5]
        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                                   transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                                 transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                             transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                             transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)  # Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        # x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        rou_fea = torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1)
        rou_fea = rou_fea.view(rou_fea.size(0), -1, rou_fea.size(3), rou_fea.size(4), rou_fea.size(5))
        rou_edema, idx_edema, logits_edema = self.router_edema(rou_fea)
        rou_ncr, idx_ncr, logits_ncr = self.router_ncr(rou_fea)
        rou_enhance, idx_enhance, logits_enhance = self.router_enhan(rou_fea)
        rou_bg, idx_bg, logits_bg = self.router_bg(rou_fea)

        multi_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multi_pos = multi_pos.view(1, 4, patch_size ** 3, transformer_basic_dims).repeat(2, 1, 1, 1)
        high_feas = (x1, x2, x3, x4)
        rou_feas = rou_fea.view(rou_fea.size(0), 4, transformer_basic_dims, rou_fea.size(2), rou_fea.size(3),
                                rou_fea.size(4))
        edema_output, de_edema = self.expert_edema(rou_edema, idx_edema, rou_feas, multi_pos, high_feas)
        ncr_output, de_ncr = self.expert_ncr(rou_ncr, idx_ncr, rou_feas, multi_pos, high_feas)
        enhan_output, de_enhan = self.expert_enhan(rou_enhance, idx_enhance, rou_feas, multi_pos, high_feas)
        bg_output, de_bg = self.expert_bg(rou_bg, idx_bg, rou_feas, multi_pos, high_feas)

        bi_output = (bg_output, ncr_output, edema_output, enhan_output)
        rou_logits = (logits_edema, logits_ncr, logits_enhance, logits_bg)

        router_weight = (rou_edema + rou_ncr + rou_enhance + rou_bg) / 4

        multi_weight = router_weight.view(x.size(0), 4, 1, 1, 1, 1)
        x5s = torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1)
        multi_feas = multi_weight * x5s  # [B,4,512,H,W,D]
        multi_feas1 = multi_feas.sum(dim=1)

        multi_feas1 = multi_feas1.view(x.size(0), -1, transformer_basic_dims)
        multi_feas1 = self.fuse_transformer(multi_feas1, self.fuse_pos)  # [B,512, H,W,D]
        multi_feas1 = multi_feas1.view(x.size(0), patch_size, patch_size, patch_size,
                                       transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        ########### InterFormer
        # flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)

        # multimodal_token_x5 = torch.cat(
        #     (flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
        #      t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
        #      t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
        #      t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
        #      ), dim=1) # [B,N ,dim]
        multimodal_token_x5 = multi_feas.view(x.size(0), -1, transformer_basic_dims)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)  # [1, 4P,C]
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multi_feas2 = multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size,
                                                     patch_size,
                                                     transformer_basic_dims * num_modals).permute(0, 4, 1, 2,
                                                                                                  3).contiguous()
        # multimodal_inter_x5 = self.multimodal_decode_conv(
        #     multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size,
        #                                    transformer_basic_dims * num_modals).permute(0, 4, 1, 2, 3).contiguous())
        multi_feas_total = torch.cat((multi_feas1, multi_feas2), dim=1)
        multimodal_inter_x5 = self.multimodal_decode_conv(multi_feas_total)
        x5_inter = multimodal_inter_x5

        fuse_pred, preds, de_x1 = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer

        if self.is_training:
            ncr_kps = self.extract_keypoints_from_masks(
                boundaries[0], bis[0],
                num_boundary_points=150,
                num_interior_points=300,
                device=x.device
            )

            edema_kps = self.extract_keypoints_from_masks(
                boundaries[1], bis[1],
                num_boundary_points=300,
                num_interior_points=600,
                device=x.device
            )

            enhance_kps = self.extract_keypoints_from_masks(
                boundaries[2], bis[2],
                num_boundary_points=120,
                num_interior_points=250,
                device=x.device
            )
            gw_loss = self.compute_gw_loss(de_x1, (de_ncr, de_edema, de_enhan), (ncr_kps, edema_kps, enhance_kps),
                                           x.device)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds, bi_output, rou_logits, gw_loss
            # return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds, bi_output, (de_x1, de_bg, de_ncr, de_edema, de_enhan)
        return fuse_pred

    def compute_gw_loss(self, de_x1, de_cats, keypoints_lists, device):

        gw_losses = []
        de_ncr, de_edema, de_enhan = de_cats
        B = de_x1.size(0)

        for cat_idx, (de_cat, kp_list) in enumerate(zip([de_ncr, de_edema, de_enhan], keypoints_lists)):
            cat_gw = []
            for b in range(B):
                kp = kp_list[b]  # [N, 3]
                if kp.numel() == 0:
                    continue

                N = kp.size(0)
                kp = kp.long().to(device)

                d_idx, h_idx, w_idx = kp[:, 0], kp[:, 1], kp[:, 2]

                sampled_de_x1 = de_x1[b:b + 1, :, h_idx, w_idx, d_idx]  # [1, C, N]
                sampled_de_cat = de_cat[b:b + 1, :, h_idx, w_idx, d_idx]

                sampled_de_x1 = sampled_de_x1.view(-1, de_x1.size(1))  # [N, C]
                sampled_de_cat = sampled_de_cat.view(-1, de_cat.size(1))

                d_x1 = torch.cdist(sampled_de_x1, sampled_de_x1)
                d_cat = torch.cdist(sampled_de_cat, sampled_de_cat)

                gw_diff = (d_x1 - d_cat) ** 2
                gw_loss_b = gw_diff.mean()

                cat_gw.append(gw_loss_b)

            if cat_gw:
                gw_losses.append(sum(cat_gw) / len(cat_gw))
            else:
                gw_losses.append(torch.tensor(0.0, device=device))

        if not gw_losses:
            return torch.tensor(0.0, device=device)

        total_gw_loss = sum(gw_losses) / len(gw_losses)
        return total_gw_loss

    def extract_keypoints_from_masks(self,
                                     boundary_mask,
                                     full_fg_mask,
                                     num_boundary_points=150,
                                     num_interior_points=300,
                                     device=None):

        if device is None:
            device = boundary_mask.device

        if boundary_mask.dim() == 5:  # [B, 1, H, W, D]
            boundary_mask = boundary_mask.squeeze(1)  # → [B, H, W, D]
        elif boundary_mask.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected boundary_mask shape: {boundary_mask.shape}")

        if full_fg_mask.dim() == 5 and full_fg_mask.size(1) == 2:
            full_fg_mask = full_fg_mask[:, 1]  # → [B, H, W, D]
        elif full_fg_mask.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected full_fg_mask shape: {full_fg_mask.shape}")

        B = boundary_mask.size(0)
        keypoints_list = []

        for b in range(B):
            bm = boundary_mask[b]  # [H, W, D]
            fgm = full_fg_mask[b]  # [H, W, D]

            boundary_coords = torch.nonzero(bm > 0.5, as_tuple=False).float()  # [N_b, 3]
            if len(boundary_coords) > num_boundary_points:
                idx = torch.randperm(len(boundary_coords), device=device)[:num_boundary_points]
                boundary_coords = boundary_coords[idx]

            interior_mask = (fgm > 0.5) & (bm <= 0.5)
            interior_coords = torch.nonzero(interior_mask, as_tuple=False).float()  # [N_i, 3]
            if len(interior_coords) > num_interior_points:
                idx = torch.randperm(len(interior_coords), device=device)[:num_interior_points]
                interior_coords = interior_coords[idx]

            if len(boundary_coords) == 0 and len(interior_coords) == 0:
                kp = torch.empty((0, 3), dtype=torch.float32, device=device)
            else:
                kp = torch.cat([boundary_coords, interior_coords], dim=0)

            keypoints_list.append(kp)

        return keypoints_list