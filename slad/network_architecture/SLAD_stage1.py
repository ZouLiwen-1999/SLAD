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

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import SwinUNETR
from slad.network_architecture.neural_network import SegmentationNetwork

class nnSwinUNETR(SegmentationNetwork):
    """
    by zlw
    """
    def __init__(self,in_channels,out_channels,img_size):#一般也会将img_size设成(96.96.96)来做
        super().__init__()
        self.swin_unetr =SwinUNETR(img_size, in_channels, out_channels, feature_size=24) # SwinUNETR VoCo的特征大小改为48
        self.conv_op = nn.Conv3d#这里是告诉nnUNet的Segmentation类我是做3D的卷积
        self.num_classes = out_channels#这里是告诉nnUNet的Segmentation类类别个数是多少
        self.do_ds=False

    def forward(self, x):
        seg_output = self.swin_unetr(x)
        if self.do_ds:
#             print('1---------------')
            return [seg_output]
        else:
#             print('2---------------')
            return seg_output

