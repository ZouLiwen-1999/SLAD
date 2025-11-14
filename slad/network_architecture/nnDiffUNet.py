# -*- coding: utf-8 -*-
# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from slad.network_architecture.unet.basic_unet import BasicUNetEncoder
from slad.network_architecture.unet.basic_unet_denose import BasicUNetDe
from slad.network_architecture.guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from slad.network_architecture.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from slad.network_architecture.guided_diffusion.resample import UniformSampler

import monai
from slad.network_architecture.neural_network import SegmentationNetwork

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiffUNet, self).__init__()
        self.embed_model = BasicUNetEncoder(3, in_channels, out_channels, [64, 64, 128, 256, 512, 64])

#         self.model = BasicUNetDe(3, in_channels + out_channels , out_channels, [64, 64, 128, 256, 512, 64], 
#                                 act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False})) #zlw 图像mask两个通道，image=x，所以一共四个通道（减去一个通道）

        self.model = BasicUNetDe(3, in_channels + out_channels-2 , out_channels, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
#             print('5555555------', x.shape)    
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, out_channels, 128,192,128), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out


class nnDiffUNet(SegmentationNetwork):
    """
    by zlw
    """
    def __init__(self,in_channels,out_channels):#
        super().__init__()
        self.diffunet = DiffUNet(in_channels,out_channels)
        self.do_ds=False
        self.conv_op = nn.Conv3d#这里是告诉nnUNet的Segmentation类我是做3D的卷积
        self.num_classes = out_channels#这里是告诉nnUNet的Segmentation类类别个数是多少

    def forward(self, x):
        image = x[:,0,:,:,:].unsqueeze(1)
        label = (x[:,1,:,:,:] + 2* x[:,2,:,:,:]).unsqueeze(1)
#         print('============',torch.unique(image),torch.unique(label))
#         raise ValueError('Just for debug!')
        
        x_t, t, noise = self.diffunet(x=label, pred_type="q_sample")
#         print('===============')
#         print('66666666------',x.shape, x_t.shape, t.shape, noise.shape)
#         print('===============')
        seg_output = self.diffunet(x=x_t, step=t, image=image, pred_type="denoise")
        
        
        if self.do_ds:
            return [seg_output]
        else:
            return seg_output

