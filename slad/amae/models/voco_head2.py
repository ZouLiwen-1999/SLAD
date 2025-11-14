import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
from utils.utils2 import mask_func
from utils.utils2 import get_mask_labels, get_mask_labelsv2
from utils.loss import * 

class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
#         print('=================',input.shape)
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)
            
#         print('-------',x.shape)
#         raise ValueError('Just for debug!')
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
class Swin2(nn.Module): #zlw模仿Hybrid里面的SwinUNETR进行改造
    def __init__(self, args):
        super(Swin2, self).__init__()
        
        self.img_size=args.img_size #zlw
        self.in_channels = args.in_channels #zlw
        downsample_size = 5 #zlw
        img_size = ensure_tuple_rep(args.img_size, args.spatial_dims) #zlw
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        
        self.select_reconstruct_region = args.select_reconstruct_region
        self.stages = []
        for i in range(downsample_size+1):
            self.stages.append((args.select_reconstruct_region[0] * 2**i, args.select_reconstruct_region[1] * 2**i))
            
        self.pretrain = args.pretrain
        if not (args.spatial_dims == 2 or args.spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")
            
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")
                    
        if not (0 <= args.drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= args.dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if args.feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
            
        self.normalize = args.normalize
        
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        #zlw 在voco投影层的基础上 加上hybrid里面的decoder
        
#         self.proj_head = projection_head(in_dim=1152, hidden_dim=2048, out_dim=2048) #feature_size=48时使用
        
        self.proj_head = projection_head(in_dim=576, hidden_dim=1024, out_dim=1024) #zlw feature_size=24时使用
        
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=8 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 8,
            out_channels=args.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 4,
            out_channels=args.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 2,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=args.spatial_dims, in_channels=args.feature_size, out_channels=args.out_channels
        )  # type: ignore
        
        if args.pretrain:
            if args.feature_size == 24:
                self.pred_mask_region = nn.Linear(384, 9)# 一个region 8个 patch
                self.contrast_learning_head = nn.Linear(384, 384)
                self.pred_mask_region_position = nn.Linear(384, 8)
            else:
                self.pred_mask_region = nn.Linear(768, 9)# 一个region 8个 patch
                self.contrast_learning_head = nn.Linear(768, 384)
                self.pred_mask_region_position = nn.Linear(768, 8)


    def wrap_feature_selection(self, feature, region_box):
        # feature: b, c, d, w, h
        return feature[..., region_box[0]:region_box[1], region_box[0]:region_box[1], region_box[0]:region_box[1]]

    def get_local_images(self, images):
        images = self.wrap_feature_selection(images, region_box=self.stages[5])
        return images
    
    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        
        b = x_in.size()[0]
        mask_func_patch_size=(24,40,40) #zlw 掩码的subpatch大小,与(4,4,4)相乘需和img_size尺寸（96,160,160）一致 !!!重要debug点
        device = x_in.device
        images = x_in.detach()
        
        if self.pretrain: #如果做预训练，需要进行随机掩码，而且得到掩码的label和position
            mask_ratio = 0.4
            x_in, mask = mask_func(x_in, self.in_channels, mask_ratio, 
                                   (16, 16, 16), #patch size
                                   (self.img_size[0]//mask_func_patch_size[0], self.img_size[1]//mask_func_patch_size[1], self.img_size[2]//mask_func_patch_size[2]) #img_size
                                  ) 
#             raise ValueError('Just for debug!')
            region_mask_labels = get_mask_labels(x_in.shape[0], 3*4*6, mask, 2*2*2, device) #3*4*6：(4x12x12)/(2x2x2)的因式分解 zlw
            region_mask_position = get_mask_labelsv2(x_in.shape[0], 3*4*6, mask, 2*2*2, device=device)
#             raise ValueError('Just for debug!')

        hidden_states_out = self.swinViT(x_in, self.normalize)
        local_images = self.get_local_images(images)
        return_x_in = self.wrap_feature_selection(x_in, region_box=self.stages[5])

        enc0 = self.encoder1(self.wrap_feature_selection(x_in, region_box=self.stages[5]))
        enc1 = self.encoder2(self.wrap_feature_selection(hidden_states_out[0], region_box=self.stages[4]))
        enc2 = self.encoder3(self.wrap_feature_selection(hidden_states_out[1], region_box=self.stages[3]))
        enc3 = self.encoder4(self.wrap_feature_selection(hidden_states_out[2], region_box=self.stages[2]))
        dec4 = self.encoder10(self.wrap_feature_selection(hidden_states_out[4], region_box=self.stages[0]))
        
        encs = [enc0, enc1, enc2, enc3, dec4] #这是voco用的
        voco_out = self.forward_encs(encs)
        voco_out = self.proj_head(voco_out.view(b, -1))
        
        dec3 = self.decoder5(dec4, self.wrap_feature_selection(hidden_states_out[3], region_box=self.stages[1]))
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        if self.pretrain:
            with torch.no_grad():
                hidden_states_out_2 = self.swinViT(x_in, self.normalize)
            encode_feature = hidden_states_out[4]
            encode_feature_2 = hidden_states_out_2[4]

            x4_reshape = encode_feature.flatten(start_dim=2, end_dim=4)
            x4_reshape = x4_reshape.transpose(1, 2)

            x4_reshape_2 = encode_feature_2.flatten(start_dim=2, end_dim=4)
            x4_reshape_2 = x4_reshape_2.transpose(1, 2)

            contrast_pred = self.contrast_learning_head(x4_reshape[:, 1])
            contrast_pred_2 = self.contrast_learning_head(x4_reshape_2[:, 1])

            pred_mask_feature = encode_feature.flatten(start_dim=2, end_dim=4)
            pred_mask_feature = pred_mask_feature.transpose(1, 2)
            mask_region_pred = self.pred_mask_region(pred_mask_feature)

            pred_mask_feature_position = encode_feature.flatten(start_dim=2, end_dim=4)
            pred_mask_feature_position = pred_mask_feature_position.transpose(1, 2)
            mask_region_position_pred = self.pred_mask_region_position(pred_mask_feature_position)

            return {
                'voco':voco_out,
                "logits": logits,
                'images': local_images,
                "pred_mask_region": mask_region_pred,
                "pred_mask_region_position": mask_region_position_pred,
                "mask": mask,
                "x_mask": return_x_in,
                "mask_position_lables": region_mask_position,
                "mask_labels": region_mask_labels,
                "contrast_pred_1": contrast_pred,
                "contrast_pred_2": contrast_pred_2,
            }
        else :
            return logits, voco_out


class VoCoHead2(nn.Module):
    def __init__(self, args):
        super(VoCoHead2, self).__init__()
        self.student = Swin2(args)
        self.teacher = Swin2(args)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        ## no scheduler here
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def forward(self, img, crops, labels):
        batch_size = labels.size()[0]
        total_size = img.size()[0] #因为在train()里面img经过了一次展平concate_image()
        sw_size = total_size // batch_size
        pos, neg, total_b_loss = 0.0, 0.0, 0.0

        img, crops = img.as_tensor(), crops.as_tensor()
        inputs = torch.cat([img, crops], dim=0)

        # here we do norm on all instances
        
        #下面是voco的损失计算------------------------------------------------------------------------------
        students_out = self.student(inputs) 
        students_all = students_out['voco'] #zlw只取logits项
        self._EMA_update_encoder_teacher() #看好了，EMA就是这么用的
        with torch.no_grad():
            teachers_all = (self.teacher(inputs)['voco']).detach() #zlw只取logits项

        x_stu_all, bases_stu_all = students_all[:total_size], students_all[total_size:]
        x_tea_all, bases_tea_all = teachers_all[:total_size], teachers_all[total_size:]

        for i in range(batch_size):
            label = labels[i]

            x_stu, bases_stu = x_stu_all[i * sw_size:(i + 1) * sw_size], bases_stu_all[i * 16:(i + 1) * 16]
            x_tea, bases_tea = x_tea_all[i * sw_size:(i + 1) * sw_size], bases_tea_all[i * 16:(i + 1) * 16]

            logits1 = online_assign(x_stu, bases_tea)
            logits2 = online_assign(x_tea, bases_stu)

            logits = (logits1 + logits2) * 0.5

#             if i == 0: #zlw 不想print
#                 print('labels and logits:', label[0].data, logits[0].data)

            pos_loss, neg_loss = ce_loss(label, logits)
            pos += pos_loss
            neg += neg_loss

            b_loss = regularization_loss(bases_stu)
            total_b_loss += b_loss

        pos, neg = pos / batch_size, neg / batch_size
        total_b_loss = total_b_loss / batch_size
        #----------------------------------------------------------------------------------------------------------
        #下面是重建项
        x_rec = torch.sigmoid(students_out['logits']) #模型预测出的重建图像
        labels = students_out['images'] #原始图像（重建GT）
        mask_images = students_out["x_mask"] #被掩码的图像
        
        loss_rec = forward_loss_reconstruct_mask(x_rec, labels, mask_images, mask_value=0.0) #重建损失

        return pos, neg, total_b_loss, loss_rec


def online_assign(feats, bases):
#     print('------feats.size():',feats.size())
#     raise ValueError('Just for debug!')
    b, c = feats.size()
    k, _ = bases.size()
    assert bases.size()[1] == c, print(feats.size(), bases.size())

    logits = []
    for i in range(b):
        feat = feats[i].unsqueeze(0)
        simi = F.cosine_similarity(feat, bases, dim=1).unsqueeze(0)
        logits.append(simi)
    logits = torch.concatenate(logits, dim=0)
    logits = F.relu(logits)

    return logits


def regularization_loss(bases):
    k, c = bases.size()
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all


def ce_loss(labels, logits):
    pos_dis = torch.abs(labels - logits)
    pos_loss = - labels * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

    neg_lab = (labels == 0).long()
    neg_loss = neg_lab * (logits ** 2)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss, neg_loss
