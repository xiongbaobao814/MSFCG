import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.fft

from layers.Embed import DataEmbedding
from layers.MSFCGBlock1 import *
from layers.RevIN import RevIN
from utils.Other import FourierLayer, series_decomp_multi

# from layers.MSGBlock import Project


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)                   # [32, 49, 7],对输入x沿维度1进行FFT变换得到频域表示
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)     # 找出前k个最高频率分量的索引
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list                 # 根据这些索引计算信号可能的周期长度period,[96, 48, 32, 24, 19]
    return period, abs(xf).mean(-1)[:, top_list]    # 对应于top_list中频率分量的平均幅度值


# 多尺度划分模块
class MutiScaleBlock(nn.Module):
    def __init__(self, configs):
        super(MutiScaleBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.k = configs.top_k

    def forward(self, x_enc):
        bs, dimension, num_nodes = x_enc.size()                     # [32, 96, 7]                
        scale_list, scale_weight = FFT_for_Period(x_enc, self.k)    # scale_weight:[32, 5]
        scale_sizes = []
        scale_nums = []
        for i in range(self.k):
            scale = scale_list[i]                                   # scale_list:[96, 48, 32, 24, 19]
            # padding
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x_enc.shape[0], (length - (self.seq_len)), x_enc.shape[2]]).to(x_enc.device)
                out = torch.cat([x_enc, padding], dim=1)
            else:
                length = self.seq_len
                out = x_enc                                          # [32, 96, 7]
            scale_num = length // scale
            out = out.reshape(bs, length // scale, scale, num_nodes) # [32, 1, 96, 7]
            scale_nums.append(scale_num)
            scale_sizes.append(out)
        return scale_list, scale_sizes, scale_nums


# 单尺度：图构建和图卷积池化模块
class GraphBlock(nn.Module):
    def __init__(self, configs):
        super(GraphBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # 仿FC_STGNN        
        self.c_out = configs.c_out                      # 7        
        self.conv_out = configs.conv_out                # 14
        self.lstmhidden_dim = configs.lstmhidden_dim    # 48
        self.lstmout_dim = configs.lstmout_dim          # 16
        self.conv_kernel = configs.conv_kernel          # 6
        self.hidden_dim = configs.hidden_dim            # 16
        self.moving_windows = configs.moving_windows    # [2, 2]
        self.strides = configs.strides                  # [1, 2]
        self.decay = configs.decay                      # 0.7
        self.pooling_choice = configs.pooling_choice    # mean
        
        # 非线性映射模块，用于特征提取
        self.nonlin_map1 = Feature_extractor_1DCNN(1, self.lstmhidden_dim, self.lstmout_dim, kernel_size=self.conv_kernel)
        self.nonlin_map2 = nn.Sequential(nn.Linear(self.lstmout_dim*self.conv_out, 2*self.hidden_dim),
                                        nn.BatchNorm1d(2*self.hidden_dim))   # 224->32
        # self.nonlin_map2 = nn.BatchNorm1d(2*self.hidden_dim)   # 224->32
        
        # # Positional Encoding
        # self.positional_encoding = PositionalEncoding2(2*self.hidden_dim, 0.1, max_len=5000)
        
        # 图构建和聚合：图卷积池化MPNN模块
        self.MPNN1 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.d_model,  
                                                moving_window=self.moving_windows[0], stride=self.strides[0], 
                                                decay=self.decay, pool_choice=self.pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.d_model, 
                                                moving_window=self.moving_windows[1], stride=self.strides[1], 
                                                decay=self.decay, pool_choice=self.pooling_choice)       
        
        # FC Graph Convolution
        self.fc = nn.Sequential(OrderedDict([   # 16x2x7,16x2 
            ('fc1', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*self.hidden_dim, self.lstmhidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(self.lstmhidden_dim, self.seq_len)),
        ]))

    def forward(self, x_enc):
        bs, scale_num, scale, num_nodes = x_enc.size()                          # [32, 1, 96, 7]                
            
        # Graph Generation
        A_input = torch.reshape(x_enc, [bs*scale_num*num_nodes, scale, 1])      # [224, 96, 1]   [448,48,1]
        A_input_map1 = self.nonlin_map1(A_input)                                # [224, 16, 14]  [448,16,8] [672, 16, 6]
        A_input_ = torch.reshape(A_input_map1, [bs*scale_num*num_nodes, -1])    # [224, 224]     [448,128] 
        
        ## Adaptive Pooling
        A_input_ = F.adaptive_avg_pool1d(A_input_.unsqueeze(1), 32).squeeze(1) # [224, 224]     [448,224]       
        A_input_map2 = self.nonlin_map2(A_input_)                               # [224, 32]      [448,32]
        A_input_ = torch.reshape(A_input_map2, [bs, scale_num, num_nodes, -1])  # [32, 1, 7, 32] [32,2,7,32] [672, 32]

        # ## positional encoding
        # X_ = torch.transpose(A_input_, 1, 2)
        # X_ = torch.reshape(X_, [bs*num_nodes, scale_num, -1])        # [224, 1, 32]    
        # X_ = self.positional_encoding(X_)
        # X_ = torch.reshape(X_, [bs, num_nodes, scale_num, -1])       # [32, 7, 1, 32] [32, 7, 2, 32]
        # X_ = torch.transpose(X_, 1, 2)
        # A_input_ = X_                                                # [32, 1, 7, 32] [32, 2, 7, 32]
        
        # Graph Convolution
        MPNN_output1 = self.MPNN1(A_input_)                           # [32, 7, 16]
        MPNN_output2 = self.MPNN2(A_input_)                           # [32, 7, 16]  # [32, 7, 32],[32, 7, 64]

        # output        
        MPNN_output = torch.cat([MPNN_output1, MPNN_output2], -1)     # [32, 7, 32]
        MPNN_output = torch.reshape(MPNN_output, [bs, num_nodes, -1]) # [32, 7, 32]
        MPNN_output = self.fc(MPNN_output)                            # [32, 7, 96]
        MPNN_output = torch.transpose(MPNN_output, 1, 2)
        
        return MPNN_output


# 多尺度图：图构建、图卷积池化和图聚合模块
class MultiScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(MultiScaleGraphBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model

        # 非线性映射模块，用于特征提取
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(GraphBlock(configs))

    def forward(self, x_enc):
        bs, dimension, num_nodes = x_enc.size()                     # [32, 96, 7]                
        
        # 多尺度提取
        scale_list, scale_weight = FFT_for_Period(x_enc, self.k)    # scale_weight:[32, 5]
        outputs = []                                                # scale_list:[96, 48, 32, 24, 19]
        for i in range(self.k):
            scale = scale_list[i] 
            ## padding
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x_enc.shape[0], (length - (self.seq_len)), x_enc.shape[2]]).to(x_enc.device)
                out = torch.cat([x_enc, padding], dim=1)
            else:
                length = self.seq_len
                out = x_enc                                          # [32, 96, 7]
            scale_num = length // scale                              # 1,2,3,4,6
            out = out.reshape(bs, scale_num, scale, num_nodes)       # [32, 1, 96, 7] [32, 2, 48, 7] [32, 3, 32, 7]
            
            ## 构造多尺度图
            output = self.gconv[i](out)         # [32, 96, 7]
            outputs.append(output)
        outputs = torch.stack(outputs, dim=-1)  # [32, 96, 7, 5]
        
        # 多尺度聚合
        scale_weight = F.softmax(scale_weight, dim=1)                                            # [32, 5]
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, dimension, num_nodes, 1) # [32, 96, 7, 5]
        outputs = torch.sum(outputs * scale_weight, -1)     # [32, 96, 7]
        
        # residual connection
        outputs = outputs + x_enc       
        return outputs
