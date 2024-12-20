import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.fft

from layers.Embed import DataEmbedding
from layers.MSFCGBlock2 import *
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
            out = self.gconv[i](out)         # [32, 96, 7]
            outputs.append(out)
            # x_enc = out
        outputs = torch.stack(outputs, dim=-1)  # [32, 96, 7, 5]
        
        # 多尺度聚合
        scale_weight = F.softmax(scale_weight, dim=1)                                            # [32, 5]
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, dimension, num_nodes, 1) # [32, 96, 7, 5]
        outputs = torch.sum(outputs * scale_weight, -1)     # [32, 96, 7]
        
        # residual connection
        outputs = outputs + x_enc       
        return outputs


# 加入patch变换后
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.freq = configs.freq
        self.embed = configs.embed
        self.num_nodes = configs.num_nodes  # 7
        self.seq_len = configs.seq_len      # 96
        self.pred_len = configs.pred_len    # 96
        self.individual = configs.individual
        self.dropout = configs.dropout      # 0.05
        self.d_model = configs.d_model      # 512
        self.layer = configs.e_layers       # 2
        self.enc_in = configs.enc_in        # 7   
        self.c_out = configs.c_out          # 7        
        
        # 1.归一化
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)
        self.revin_fc = nn.Linear(in_features=self.num_nodes, out_features=self.d_model)
        
        # 时间序列趋势性和季节性分解
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])
        self.seasonality_model = FourierLayer(pred_len=0, k=5)
        
        # 2.编码
        self.enc_embedding = DataEmbedding(self.num_nodes, self.d_model, self.embed, self.freq, self.dropout)
        self.embed_fc = nn.Linear(self.d_model, self.c_out, bias=True)
        
        # 3.多尺度图模块
        self.model = nn.ModuleList([MultiScaleGraphBlock(configs) for _ in range(self.layer)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.projection = nn.Linear(self.enc_in, self.c_out, bias=True)
        self.projection1 = nn.Linear(self.d_model, self.num_nodes, bias=True)
        self.seq2pred = Project(self.individual, self.c_out, self.seq_len, self.pred_len, self.dropout)

    # 时间序列趋势性和季节性分解
    def seasonality_and_trend_decompose(self, x):
        _, trend = self.trend_model(x)              # [64, 96, 7]
        seasonality, _ = self.seasonality_model(x)  # [64, 96, 7]
        return x + seasonality + trend

    def forward(self, x_enc, x_mark_enc, mask=None):    # [32, 96, 7], [32, 96, 4]
        
        # 1.归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # if self.revin:
        #     x_enc = self.revin_layer(x_enc, 'norm')       # [32, 96, 7]
        # # x_enc = self.revin_fc(x_enc)                    # [32, 96, 512]
        
        # x_enc = self.seasonality_and_trend_decompose(x_enc)   # [32, 96, 7]
        
        # 2.编码
        x_enc = self.enc_embedding(x_enc, x_mark_enc)     # [32, 96, 512]
        # x_input = self.embed_fc(x_input)                    # [32, 96, 7]

        # 3.多尺度图处理, 包含layer个MultiScaleGraphBlock实例
        for i in range(self.layer):   # 2层
            x_enc = self.seasonality_and_trend_decompose(x_enc)
            x_enc = self.enc_embedding(x_enc, x_mark_enc)
            layer_out = self.model[i](x_enc)              # [32, 96, 7]
            x_input = self.layer_norm(layer_out)            # [32, 96, 7], 归一化
        
        # 4.投影
        out = self.projection1(x_input)                           # [32, 96, 7]        
        out = self.seq2pred(out.transpose(1, 2)).transpose(1, 2)  # [32, 96, 7]

        # 5.De-Normalization from Non-stationary Transformer
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # if self.revin:
        #     out = self.revin_layer(out, 'denorm')
            
        return out[:, -self.pred_len:, :]      # [32, 96, 7]
    