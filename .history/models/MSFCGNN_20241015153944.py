import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.fft

from layers.Embed import DataEmbedding
from layers.MSFCGBlock import *
from layers.RevIN import RevIN


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)                   # [32, 49, 512],对输入x沿维度1进行FFT变换得到频域表示
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)     # 找出前k个最高频率分量的索引
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list                 # 根据这些索引计算信号可能的周期长度period,[96, 48, 32, 24, 19]
    return period, abs(xf).mean(-1)[:, top_list]    # 对应于top_list中频率分量的平均幅度值


class MutiScaleBlock(nn.Module):
    def __init__(self, configs):
        super(MutiScaleBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        
        # 仿FC_STGNN        
        self.conv_out = configs.conv_out
        self.lstmhidden_dim = configs.lstmhidden_dim
        self.lstmout_dim = configs.lstmout_dim
        self.conv_kernel = configs.conv_kernel
        self.hidden_dim = configs.hidden_dim
        self.time_length = configs.time_denpen_len
        self.num_windows = configs.num_windows
        self.moving_windows = configs.moving_windows
        self.stride = configs.stride
        self.decay = configs.decay
        self.pooling_choice = configs.pooling_choice

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
    
    
class GraphBlock(nn.Module):
    def __init__(self, configs):
        super(GraphBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # 仿FC_STGNN        
        self.conv_out = configs.conv_out
        self.lstmhidden_dim = configs.lstmhidden_dim
        self.lstmout_dim = configs.lstmout_dim
        self.conv_kernel = configs.conv_kernel
        self.hidden_dim = configs.hidden_dim
        self.time_length = configs.time_denpen_len
        self.num_windows = configs.num_windows
        self.moving_windows = configs.moving_windows
        self.stride = configs.stride
        self.decay = configs.decay
        self.pooling_choice = configs.pooling_choice
        
        # 非线性映射模块，用于特征提取
        self.nonlin_map = Feature_extractor_1DCNN(1, self.lstmhidden_dim, self.lstmout_dim, kernel_size=self.conv_kernel)
        self.nonlin_map2 = nn.Sequential(nn.Linear(self.lstmout_dim*self.conv_out, 2*self.hidden_dim),
                                        nn.BatchNorm1d(2*self.hidden_dim))   # 224->32
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(2*self.hidden_dim, 0.1, max_len=5000)
        
        # 图构建和聚合：图卷积池化MPNN模块
        self.MPNN1 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
                                                moving_window=self.moving_windows[0], stride=self.stride[0], 
                                                decay=self.decay, pool_choice=self.pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
                                                moving_window=self.moving_windows[1], stride=self.stride[1], 
                                                decay=self.decay, pool_choice=self.pooling_choice)       
        # FC Graph Convolution
        self.fc = nn.Sequential(OrderedDict([   # 16x2x7,16x2 
            ('fc1', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*self.hidden_dim, self.seq_len)),
        ]))

    def forward(self, x_enc):
        bs, scale_num, scale, num_nodes = x_enc.size()                          # [32, 1, 96, 7]                
            
        ## Graph Generation
        A_input = torch.reshape(x_enc, [bs*scale_num*num_nodes, scale, 1])      # [224, 96, 1]   [448,48,1]
        A_input_map1 = self.nonlin_map(A_input)                                 # [224, 16, 14]  [448,16,8]
        A_input_ = torch.reshape(A_input_map1, [bs*scale_num*num_nodes, -1])    # [224, 224]     [448,128] 
        
        # Adaptive Pooling
        A_input_ = F.adaptive_avg_pool1d(A_input_.unsqueeze(1), 224).squeeze(1) # [224, 224]     [448,224]       
        A_input_map2 = self.nonlin_map2(A_input_)                               # [224, 32]      [448,32]
        A_input_ = torch.reshape(A_input_map2, [bs, scale_num, num_nodes, -1])  # [32, 1, 7, 32] [32,2,7,32]

        ## positional encoding
        X_ = torch.reshape(A_input_, [bs, scale_num, num_nodes, -1]) # [32, 1, 7, 32] [32, 2, 7, 32]
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs*num_nodes, scale_num, -1])        # [224, 1, 32]    
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_nodes, scale_num, -1])       # [32, 7, 1, 32]  [32, 7, 2, 32]
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_                                                # [32, 1, 7, 32] [32, 2, 7, 32]

        ## Graph Convolution
        MPNN_output1 = self.MPNN1(A_input_)                          # [32, 1, 7, 16]
        MPNN_output2 = self.MPNN2(A_input_)                          # [32, 1, 7, 16]

        ## output        
        feature = torch.cat([MPNN_output1, MPNN_output2], -1)       # [32, 1, 7, 32]
        feature = torch.reshape(feature, [bs, num_nodes, -1])       # [32, 7, 32]        
        feature = self.fc(feature)                                  # [32, 7, 96]
        feature = torch.transpose(feature, 1, 2)
        return feature


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()        
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        
        # 仿FC_STGNN        
        self.conv_out = configs.conv_out
        self.lstmhidden_dim = configs.lstmhidden_dim
        self.lstmout_dim = configs.lstmout_dim
        self.conv_kernel = configs.conv_kernel
        self.hidden_dim = configs.hidden_dim
        self.time_length = configs.time_denpen_len
        self.num_windows = configs.num_windows
        self.moving_windows = configs.moving_windows
        self.stride = configs.stride
        self.decay = configs.decay
        self.pooling_choice = configs.pooling_choice

        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()        
        # 非线性映射模块，用于特征提取
        self.gconv = nn.ModuleList()
        for i in range(self.k-1):
            self.gconv.append(GraphBlock(configs))

    def forward(self, x_enc):
        bs, dimension, num_nodes = x_enc.size()                     # [32, 96, 7]                
        scale_list, scale_weight = FFT_for_Period(x_enc, self.k)    # scale_weight:[32, 5]
        outputs = []                                                # scale_list:[96, 48, 32, 24, 19]
        for i in range(self.k-1):
            scale = scale_list[i+1] 
            # padding
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x_enc.shape[0], (length - (self.seq_len)), x_enc.shape[2]]).to(x_enc.device)
                out = torch.cat([x_enc, padding], dim=1)
            else:
                length = self.seq_len
                out = x_enc                                          # [32, 96, 7]
            scale_num = length // scale                              # 1
            out = out.reshape(bs, scale_num, scale, num_nodes)       # [32, 2, 48, 7] [32, 3, 32, 7]
            
            # 构造多尺度图
            output = self.gconv[i](out)     # [32, 96, 7]
            outputs.append(output)            
        return outputs


# 加入patch变换后
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        
        # 仿FC_STGNN        
        self.conv_out = configs.conv_out
        self.lstmhidden_dim = configs.lstmhidden_dim
        self.lstmout_dim = configs.lstmout_dim
        self.conv_kernel = configs.conv_kernel
        self.hidden_dim = configs.hidden_dim
        self.time_length = configs.time_denpen_len
        self.num_windows = configs.num_windows
        self.moving_windows = configs.moving_windows
        self.stride = configs.stride
        self.decay = configs.decay
        self.pooling_choice = configs.pooling_choice
        self.n_class = configs.n_class        
                
        # # 仿MSGNet逻辑
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # self.layer = configs.e_layers
        # self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        # self.layer_norm = nn.LayerNorm(configs.d_model)
        # self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        # self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # # self.seq2pred = Predict(configs.individual ,configs.c_out,configs.seq_len, configs.pred_len, configs.dropout)
        
        # 归一化
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        
        # 多周期尺度提取
        self.scaleset = MutiScaleBlock(configs)
        self.scalegraph = ScaleGraphBlock(configs)
        
        # # 非线性映射模块，用于特征提取
        # self.nonlin_map = Feature_extractor_1DCNN(1, self.lstmhidden_dim, self.lstmout_dim, kernel_size=self.conv_kernel)
        # self.nonlin_map2 = nn.Sequential(nn.Linear(self.lstmout_dim*self.conv_out, 2*self.hidden_dim),
        #                                 nn.BatchNorm1d(2*self.hidden_dim))   # 128->32
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # # Positional Encoding
        # self.positional_encoding = PositionalEncoding(2*self.hidden_dim, 0.1, max_len=5000)
        
        # # 图构建和聚合：图卷积池化MPNN模块
        # self.MPNN1 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
        #                                         moving_window=self.moving_windows[0], stride=self.stride[0], 
        #                                         decay=self.decay, pool_choice=self.pooling_choice)
        # self.MPNN2 = GraphConvpoolMPNN_block(2*self.hidden_dim, self.hidden_dim, self.num_nodes, self.time_length, 
        #                                         moving_window=self.moving_windows[1], stride=self.stride[1], 
        #                                         decay=self.decay, pool_choice=self.pooling_choice)       
        # # FC Graph Convolution
        # self.fc1 = nn.Sequential(OrderedDict([   # 16x2x7,16x2 
        #     ('fc1', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('fc2', nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('fc3', nn.Linear(2*self.hidden_dim, self.seq_len)),
        # ]))
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        bs, dimension, num_nodes = x_enc.size()                        # [100, 2, 9, 64]  [32, 96, 7]                

        outputs = self.scalegraph(x_enc)
        print('outputs shape is {}'.format(outputs.shape))
        return outputs
    
        # for i in range(self.layer):   # 2层,共包含2个ScaleGraphBlock实例
        #     layer_out = self.model[i](enc_out)     # [32, 96, 512]
        #     enc_out = self.layer_norm(layer_out)   # [32, 96, 512],归一化
    
        # ## Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc = x_enc / stdev
        
        # if self.revin:
        #     x_enc = self.revin_layer(x_enc, 'norm')     # [64, 96, 21]
        
#         ## muti-scale
#         scale_list, scale_sizes, scale_nums = self.scaleset(x_enc)
#         i = 1
#         scale =scale_list[i]        # 48
#         A_input = scale_sizes[i]    # [32, 2, 48, 7]
#         scale_num = scale_nums[i]   # 2
        
#         ## Graph Generation
#         A_input = torch.reshape(A_input, [bs*scale_num*num_nodes, scale, 1]) # [224, 96, 1]   [448,48,1]
#         A_input_map1 = self.nonlin_map(A_input)                                  # [224, 18, 14]  [448,16,8]
#         A_input_ = torch.reshape(A_input_map1, [bs*scale_num*num_nodes, -1])     # [224, 252]     [448,128]        
        
#         A_input_ = F.adaptive_avg_pool1d(A_input_.unsqueeze(1), 128).squeeze(1)
#         A_input_map2  = self.nonlin_map2(A_input_)                                # [224, 32]      [448,32]
#         A_input_ = torch.reshape(A_input_map2, [bs, scale_num, num_nodes, -1])   # [32, 1, 7, 32] [32,2,7,32]
# # 672x96 and 128x32

#         ## positional encoding
#         X_ = torch.reshape(A_input_, [bs, scale_num, num_nodes, -1]) # [32, 1, 7, 32] [32, 2, 7, 32]
#         X_ = torch.transpose(X_, 1, 2)
#         X_ = torch.reshape(X_, [bs*num_nodes, scale_num, -1])        # [224, 1, 32]    
#         X_ = self.positional_encoding(X_)
#         X_ = torch.reshape(X_, [bs, num_nodes, scale_num, -1])       # [32, 7, 1, 32]  [32, 7, 2, 32]
#         X_ = torch.transpose(X_, 1, 2)
#         A_input_ = X_                                                # [32, 1, 7, 32] [32, 2, 7, 32]

#         ## Graph Convolution
#         MPNN_output1 = self.MPNN1(A_input_)                          # [32, 1, 7, 16]
#         MPNN_output2 = self.MPNN2(A_input_)                          # [32, 1, 7, 16]
        
#         features = torch.cat([MPNN_output1, MPNN_output2], -1)       # [32, 1, 7, 32]
#         features = torch.reshape(features, [bs, num_nodes, -1])      # [32, 7, 32]        
#         features = self.fc1(features)                                # [32, 7, 96]
#         features = torch.transpose(features, 1, 2) 
#         return features
    
#     def forward_V1(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         bs, dimension, num_nodes = x_enc.size()                        # [100, 2, 9, 64]  [32, 96, 7]                

#         # ## Normalization from Non-stationary Transformer
#         # means = x_enc.mean(1, keepdim=True).detach()
#         # x_enc = x_enc - means
#         # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         # x_enc1 = x_enc / stdev
        
#         ## norm
#         # if self.revin:
#         #     x_enc = self.revin_layer(x_enc, 'norm')     # [64, 96, 21]
        
#         ## muti-scale
#         scale_list, scale_sizes, scale_nums = self.scaleset(x_enc)
#         scale =scale_list[1]
#         A_input = scale_sizes[1]    # [32, 2, 48, 7]
#         scale_num = scale_nums[1]   # 2
        
#         ## Graph Generation
#         A_input = torch.reshape(A_input, [bs*scale_num*num_nodes, scale, 1]) # [1800, 64, 1]   [224, 96, 1]  [448,48,1]
#         A_input_ = self.nonlin_map(A_input)                                  # [1800, 18, 10]  [224, 18, 14] [448,18,8]
#         A_input_ = torch.reshape(A_input_, [bs*scale_num*num_nodes, -1])     # [1800, 180]     [224, 252]    [448,144]        
#         A_input_ = self.nonlin_map2(A_input_)                                # [1800, 32]      [224, 32]     [448,32]
#         A_input_ = torch.reshape(A_input_, [bs, scale_num, num_nodes, -1])   # [100, 2, 9, 32] [32, 1, 7, 32][32,2,7,32]
# # test loss句时报错：672x108 and 144x32

#         ## positional encoding
#         X_ = torch.reshape(A_input_, [bs, scale_num, num_nodes, -1]) # [100, 2, 9, 32]  [32, 1, 7, 32]  [32, 2, 7, 32]
#         X_ = torch.transpose(X_, 1, 2)
#         X_ = torch.reshape(X_, [bs*num_nodes, scale_num, -1])        # [900, 2, 32]     [224, 1, 32]    
#         X_ = self.positional_encoding(X_)
#         X_ = torch.reshape(X_, [bs, num_nodes, scale_num, -1])       # [100, 9, 2, 32]  [32, 7, 1, 32]
#         X_ = torch.transpose(X_, 1, 2)
#         A_input_ = X_                                                # [100, 2, 9, 32]  [32, 1, 7, 32]  [32, 2, 7, 32]

#         ## Graph Convolution
#         MPNN_output1 = self.MPNN1(A_input_)                          # [100, 1, 9, 16]  [32, 1, 7, 16]
#         MPNN_output2 = self.MPNN2(A_input_)                          # [100, 1, 9, 16]  [32, 1, 7, 16]

#         # features1 = torch.reshape(MPNN_output1, [bs, -1])            # [100, 144]   [32, 112]
#         # features2 = torch.reshape(MPNN_output2, [bs, -1])            # [100, 144]   [32, 112]
#         # features = torch.cat([features1,features2], -1)              # [100, 288]   [32, 224]        
#         # features = self.fc(features)                                 # [100, 6]     [32, 6]
        
#         features = torch.cat([MPNN_output1, MPNN_output2], -1)        # [32, 1, 7, 32]
#         features = torch.reshape(features, [bs, num_nodes, -1])       # [32, 7, 32]        
#         features = self.fc1(features)                                 # [32, 7, 96]
#         features = torch.transpose(features, 1, 2) 
#         return features
