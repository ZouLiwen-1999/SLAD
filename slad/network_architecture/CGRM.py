# -*- coding: utf-8 -*-
'''
输入图像特征和目标分割logit，构建对应的state特征矩阵、投影矩阵和反投影矩阵
By Liwen Zou
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm = nn.BatchNorm3d

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("input x in GCN can not be NaN!")
            
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        if torch.isnan(h).any():
            raise ValueError("h1 in GCN can not be NaN!")
            
        h = h - x
        
        if torch.isnan(h).any():
            raise ValueError("h2 in GCN can not be NaN!")
            
        h = self.relu(self.conv2(h))
        
        if torch.isnan(h).any():
            raise ValueError("h3 in GCN can not be NaN!")
            
        return h


class GraphGenerator(nn.Module):
    def __init__(self, num_in, plane_mid, mids, abn=BatchNorm, normalize=False):
        '''
        Graph Reasoning Module (GRM)
        输入l(lession), o(organ normal part)特征，o与l信息交互形成投影矩阵B作用于l，将l特征投影到交互空间
        参数：
        num_in：
        plane_mid：
        mids：
        '''
        super(GraphGenerator, self).__init__()

#         self.normalize = normalize
        self.num_s = int(plane_mid) #每个节点的特征维数
        self.num_n = (mids) * (mids) #图节点个数
        
        self.priors = nn.AdaptiveAvgPool3d(output_size=(mids + 2, mids + 2, mids + 2))

        self.conv_state = nn.Conv3d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv3d(num_in, self.num_s, kernel_size=1)
#         self.conv_extend = nn.Conv3d(self.num_s, num_in, kernel_size=1, bias=False)
#         self.gcn = GCN(num_state=self.num_s, num_node=self.num_n) 
#         self.blocker = abn(num_in)


    def forward(self, x, y): 
        n, c, h, w, d =  x.size() #获取l特征的尺寸
#         print('In CGRM==========Input x size:',x.size(),'y size:',y.size())
        #构建投影矩阵
        #print('########self.num_s',self.num_s)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)#将x卷积到state空间并降维
        #print('########x_state_reshaped',x_state_reshaped.size())
        x_proj = self.conv_proj(x)#将x投影到state空间
#         print('########x_proj',x_proj.size())

        y_prob = F.interpolate(y.unsqueeze(1), size=(h, w, d), mode='nearest') #将y上采样到和x一样尺寸(h x w)
    
        x_mask = x_proj * y_prob #x与y概率图点乘得到权重图
        #print('########x_mask',x_mask.size())
        
        #print('#######self.num_s',self.num_s)
        #print('########self.priors(x_mask)',self.priors(x_mask).size())
        x_aohor = self.priors(x_mask)[:,:,1:-1,1:-1,1:-1].reshape(n, self.num_s, -1) #对权重图池化
        
        #print('#######x_aohor.permute(0, 2, 1)', x_aohor.permute(0, 2, 1).size())
        #print('#######x_proj.reshape(n, self.num_s, -1)', x_proj.reshape(n, self.num_s, -1).permute(0, 2, 1).size())
        x_proj_reshaped = torch.matmul(x_aohor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1)) #矩阵乘法得到相似性
        #print('#######x_proj_reshaped11111', x_proj_reshaped.size())
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1) #通过softmax得到attention
        #print('#######x_proj_reshaped2222', x_proj_reshaped.size())
        x_rproj_reshaped = x_proj_reshaped #反投影矩阵定义为投影矩阵一致 不定义为转置？
        #print('#######x_proj_reshaped3333', x_proj_reshaped.size())
        
        return x_state_reshaped,x_proj_reshaped,x_rproj_reshaped
    
class CRGM(nn.Module):
    def __init__(num_c, num_s,num_n_per_dim, normalize=False, abn=BatchNorm):
        self.num_c = num_c #输入通道数, 需要根据输入特征图尺度确定
        self.nums = num_s #每个图节点的特征维数
        self.num_n_per_dim = num_n_per_dim #num_n_per_dim * num_n_per_dim * num_n_per_dim 代表一个三维图像的节点个数
        self.num_n = 2*num_n_per_dim*num_n_per_dim*num_n_per_dim
        self.normalize = normalize
        self.abn = abn
        self.gcn = GCN(num_s,self.num_n)
        self.gg = GraphGenerator(num_c, num_s, num_n_per_dim, abn)#定义图和投影矩阵生成器
        self.blocker = abn(num_c//2)
        self.conv_extend = nn.Conv3d(num_s, num_c//2, kernel_size=1, bias=False)
        super(CRGM, self).__init__()
        
    def forward(self, x, organ, tumor):
        n, c, h, w, d =  x.size() #获取特征图的尺寸
#         print('==========Input size:',x.size(),'data type:',x.dtype)
        o_state_reshaped,o_proj_reshaped,o_rproj_reshaped = self.gg(x, organ.detach()) 
#         print('1-----------',o_state_reshaped.shape,o_proj_reshaped.shape)
        t_state_reshaped,t_proj_reshaped,t_rproj_reshaped = self.gg(x, tumor.detach()) 
#         print('2-----------',o_state_reshaped.shape,o_proj_reshaped.shape)
        
        o_n_state = torch.matmul(o_state_reshaped, o_proj_reshaped.permute(0, 2, 1)) #投影
#         print('3-----------',o_n_state.shape)
        t_n_state = torch.matmul(t_state_reshaped, t_proj_reshaped.permute(0, 2, 1)) #投影
        
        ot_n_state=torch.cat([o_n_state,t_n_state], dim=2) #将两组节点串联 串联的维度待考虑？
#         print('4-----------',ot_n_state.shape)

        if self.normalize:
            ot_n_state = ot_n_state * (1. / o_state_reshaped.size(2)) #哪个维度归一化？
                
        ot_n_rel = self.gcn(ot_n_state) #实现图推理
    
        if torch.isnan(ot_n_rel).any():
            raise ValueError("ot_n_rel can not be NaN!")
        
        o_n_rel=ot_n_rel[:,:,self.num_n//2:] #将修正后的organ特征提取出来
        t_n_rel=ot_n_rel[:,:,:self.num_n//2] #将修正后的tumor特征提取出来

        # 反投影
        o_state_reshaped = torch.matmul(o_n_rel, o_rproj_reshaped)       #x_n_rel   ###没有gcn
        o_state = o_state_reshaped.view(n, self.num_s, *x.size()[2:]) #将修正后的特征reshape成原来的大小
        
#         if torch.isnan(o_state_reshaped).any():
#             raise ValueError("o_state_reshaped can not be NaN!")
            
#         if torch.isnan(o_state).any():
#             raise ValueError("o_state can not be NaN!")
        
        t_state_reshaped = torch.matmul(t_n_rel, t_rproj_reshaped)       #x_n_rel   ###没有gcn
        t_state = t_state_reshaped.view(n, self.num_s, *x.size()[2:]) #将修正后的特征reshape成原来的大小
        
        
        out_o =  self.blocker(self.conv_extend(o_state))
        out_t =  self.blocker(self.conv_extend(t_state))
        
#         print('5-----------',out_o.shape)
#         print('6-----------',out_t.shape)
        
        out = x + torch.cat([out_o,out_t],dim=1)
        
#         print('7-----------',x.shape)
#         raise ValueError("CGRM debug .....")
        
        return out
        
    
    






