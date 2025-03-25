import math
import torch
import torch.nn as nn
from models.resnet import Resnet1D
from models.vq_model_dual import Normalize, nonlinearity

class PositionalEncoding(nn.Module):
    def __init__(self, src_dim, embed_dim, dropout, max_len=100, hid_dim=512):
        """
        :param src_dim:  orignal input dimension
        :param embed_dim: embedding dimension
        :param dropout: dropout rate
        :param max_len: max length
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(src_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, embed_dim)
        self.relu = nn.ReLU()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / embed_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, step=None):
        """
        :param input: L x N x D
        :param step:
        :return:
        """
        # raw_shape = input.shape[:-2]
        # j_num, f_dim = input.shape[-2], input.shape[-1]
        # input = input.reshape(-1, j_num, f_dim).transpose(0, 1)
        emb = self.linear2(self.relu(self.linear1(input)))
        emb = emb * math.sqrt(self.embed_dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        # emb = emb.transpose(0, 1).reshape(raw_shape + (j_num, -1))
        return emb

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, local_feat, global_feat):
        """
        Args:
            local_feat: [T, B, C] 解码器当前特征
            global_feat: [T', B, C] Transformer全局特征
        """
        # 维度对齐
        if local_feat.shape[0] != global_feat.shape[0]:
            global_feat = F.interpolate(
                global_feat.permute(1,2,0), 
                size=local_feat.shape[0],
                mode='linear'
            ).permute(2,0,1)
            
        # 注意力计算
        local_feat = self.norm(local_feat)
        attn_out, _ = self.attn(
            query=local_feat,
            key=global_feat,
            value=global_feat
        )
        return local_feat + attn_out  # 残差连接

class EnhancedDecoder(nn.Module):
    def __init__(self, d_model=256,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 num_attention_heads=8):
        super().__init__()
        
        # 初始投影层
        self.init_conv = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )
        
        self.global_conv = nn.Sequential(
            nn.Conv1d(d_model, width, 3, 1, 1),
            nn.ReLU()
        )
        # 构建解码块
        self.blocks = nn.ModuleList()
        for i in range(down_t):
            block = nn.ModuleDict({
                'resnet': Resnet1D(width, depth, dilation_growth_rate, 
                                 reverse_dilation=True, 
                                 activation=activation, 
                                 norm=norm),
                'upsample': nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(width, width, 3, 1, 1)
                ),
                'attention': CrossAttentionLayer(width, num_attention_heads)
            })
            self.blocks.append(block)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )

    def forward(self, x, global_features):
        """
        Args:
            x: 编码器输出特征 [B, C, T]
            global_features: Transformer全局特征 [B, T', C]
        """
        x = self.init_conv(x)  # [B, W, T]
        
        # 转换全局特征维度
        global_features = self.global_conv(global_features).permute(2, 0, 1)  # [T', B, C]
        
        for block in self.blocks:
            # 残差网络处理
            x = block['resnet'](x)
            
            # 上采样
            x = block['upsample'](x)
            
            # 交叉注意力
            x = x.permute(2, 0, 1)  # [T, B, C]
            x = block['attention'](x, global_features)
            x = x.permute(1, 2, 0)  # 恢复[B, C, T]
            
        x = self.final_conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 with_attn=False):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            if with_attn:
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    MotionAttention(width, num_heads=8),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(width, out_dim, 3, 1, 1)
                )
            else:
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(width, out_dim, 3, 1, 1)
                )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
# class Decoder_wo_upsample(nn.Module):
#     def __init__(self,
#                  input_emb_width = 3,
#                  output_emb_width = 512,
#                  down_t = 3,
#                  stride_t = 2,
#                  width = 512,
#                  depth = 3,
#                  dilation_growth_rate = 3, 
#                  activation='relu',
#                  norm=None,
#                  with_attn=False):
#         super().__init__()
#         blocks = []
        
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         self.with_attn = with_attn
#         blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         for i in range(down_t):
#             out_dim = width
#             if self.with_attn:
#                 block = nn.Sequential(
#                     MotionAttention(width, num_heads=8),
#                     Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                     nn.Conv1d(width, out_dim, 3, 1, 1)
#                 )
#             else:
#                 block = nn.Sequential(
#                     Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                     nn.Conv1d(width, out_dim, 3, 1, 1)
#                 )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         return self.model(x)

class Decoder_wo_upsample(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 with_attn=False):
        super().__init__()
        blocks = []
        self.with_attn = with_attn
        # if with_attn:
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            if self.with_attn:
                block = nn.ModuleDict({
                    'attention': MotionAttention(width, num_heads=8),
                    'resnet': Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    'conv': nn.Conv1d(width, out_dim, 3, 1, 1)
                })
            else:
                block = nn.ModuleDict({
                    'resnet': Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    'conv': nn.Conv1d(width, out_dim, 3, 1, 1)
                })
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)
        # else:
        #     filter_t, pad_t = stride_t * 2, stride_t // 2
            
        #     blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        #     blocks.append(nn.ReLU())
        #     for i in range(down_t):
        #         out_dim = width
        #         if self.with_attn:
        #             block = nn.Sequential(
        #                 MotionAttention(width, num_heads=8),
        #                 Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
        #                 nn.Conv1d(width, out_dim, 3, 1, 1)
        #             )
        #         else:
        #             block = nn.Sequential(
        #                 Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
        #                 nn.Conv1d(width, out_dim, 3, 1, 1)
        #             )
        #         blocks.append(block)
        #     blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        #     blocks.append(nn.ReLU())
        #     blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        #     self.model = nn.Sequential(*blocks)
            
    def forward(self, x, motion_mask=None):
        """ 
        Args:
            x: 输入特征 [B, C, T]
            motion_mask: 注意力掩码 [B, 1, T] 或 None
        """
        # if self.with_attn:
        x = self.blocks[0](x)  # 第一层卷积
        x = self.blocks[1](x)  # 第一层激活
        for block in self.blocks[2:-3]:  # 遍历解码块
            if 'attention' in block:
                x = block['attention'](x, motion_mask)  # 传递 motion_mask
            x = block['resnet'](x)
            x = block['conv'](x)
        x = self.blocks[-3](x)  # 倒数第二层卷积
        x = self.blocks[-2](x)  
        x = self.blocks[-1](x)  # 最后一层卷积
        return x
        # else:
        #     return self.model(x)
        

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class MotionAttention(nn.Module):
    """运动序列注意力机制"""
    def __init__(self, dim, num_heads=4, residual=True):
        super().__init__()
        self.residual = residual
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        # self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, motion_mask = None):
        x = x.permute(0, 2, 1)  # (B, D, L)
        B, L, D = x.shape
        residual_x = x
        if motion_mask is not None:
            motion_mask = motion_mask.permute(0, 2, 1)
            x = self.attn(x, x, x, key_padding_mask=motion_mask)[0]
        else:
            x = self.attn(x, x, x)[0]
        if self.residual:
            result = self.proj(x) + residual_x
        else:
            result = self.proj(x)
        return result.permute(0, 2, 1)


class MotionDecoder(nn.Module):
    def __init__(self, 
                 input_dim=256,
                 output_dim=45,  # 例如人体关节的3D坐标
                 seq_len=196,     # 基础序列长度
                 dim=512,
                 num_layers=6,
                 num_heads=8,
                 expansion=4,
                 dropout=0.1):
        super().__init__()
        
        # 初始投影
        self.init_proj = nn.Linear(input_dim, dim)
        
        # 时间位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        
        # 解码层堆叠
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "res_block": TemporalResBlock(dim, expansion, dropout),
                "attn": MotionAttention(dim, num_heads),
                "norm": nn.LayerNorm(dim)
            }) for _ in range(num_layers)
        ])
        
        # 输出层
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 (B, L, D_in)
        Returns:
            out: 解码序列 (B, L, D_out)
        """
        B, L, _ = x.shape
        
        # 初始投影
        x = self.init_proj(x)  # (B, L, D)
        
        # 添加位置编码
        x += self.pos_embed[:, :L]
        
        # 通过各层
        for layer in self.layers:
            # 残差连接
            residual = x
            x = layer["res_block"](x)
            x = layer["attn"](x) + residual
            
            # 层归一化
            x = layer["norm"](x)
        
        # 最终输出
        x = self.final_norm(x)
        return self.output_proj(x)
    
class PureMotionDecoder(nn.Module):
    def __init__(self, output_dim, d_model=256, num_layers=2, num_parts=6, with_attn=False, with_global = True):
        super().__init__()
        self.with_global = with_global
        if with_global:
            # 全局时空编码器
            self.global_encoder = nn.Sequential(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
                    num_layers=num_layers
                ),
                TemporalConvBlock(d_model)  # 新增时序卷积模块
            )
        
        # 分部位解码器
        self.part_decoders = nn.ModuleList([
            PartDecoderV2(d_model, num_layers) for _ in range(num_parts)
        ])
        
        self.dim_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, output_dim[i])
            ) for i in range(num_parts)
        ])
        if with_attn and with_global:
            self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
            for _ in range(num_parts)
        ])
        else:
        # 特征融合门控
            self.fusion_gate = nn.Parameter(torch.ones(num_parts))
        self.with_attn = with_attn
    def forward(self, motion_features):
        """
        Input: motion_features [bs,7,seq_len,dim] 
               (包含1个全局特征+6个部位特征)
        Output: [bs,6,seq_len,dim]
        """
        # 全局特征增强
        global_feat = motion_features[0].permute(0,2,1)  # [bs,dim,seq_len]
        # global_feat = motion_features[:,0]  # [bs,seq_len,dim]
        if self.with_global:
            global_encoded = self.global_encoder(global_feat)  # [bs,seq_len,dim]
        
        # 门控特征融合
        reconstructions = []
        for i in range(6):
            part_feat = motion_features[i+1].permute(0,2,1)
            if self.with_global:
                if self.with_attn:
                    # 交叉注意力融合
                    attn_output, _ = self.cross_attns[i](
                        query=part_feat,
                        key=global_encoded,
                        value=global_encoded
                    )
                    # 残差连接
                    fused = part_feat + attn_output
                else:
                    # 动态门控融合
                    gate = torch.sigmoid(self.fusion_gate[i])
                    fused = gate * global_encoded + (1-gate) * part_feat
            else:
                fused = part_feat
            # 部位解码
            decoded = self.part_decoders[i](fused)
            reconstructions.append(decoded)
        result = [adapter(reconstructions[i]) for i, adapter in enumerate(self.dim_adapter)]
        return result

class PartDecoderV2(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.time_blocks = nn.Sequential(
            *[TemporalResBlock(dim) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        return self.time_blocks(x)

class TemporalResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=4,
            batch_first=True
        )
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=3, 
            padding=1, groups=4  # 分组卷积提升效率
        )
        
    def forward(self, x):
        # 分支1：局部注意力
        attn_out, _ = self.attn(x, x, x)
        # 分支2：深度卷积
        conv_out = self.conv(x.permute(0,2,1)).permute(0,2,1)
        # 残差融合
        return x + attn_out + conv_out

class TemporalConvBlock(nn.Module):
    """多尺度时序特征提取"""
    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim//4, kernel_size, padding=kernel_size//2)
            for kernel_size in [3, 5, 7, 9]
        ])
        self.fuse = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [bs, seq_len, dim]
        x_t = x.permute(0,2,1)  # [bs, dim, seq_len]
        features = [conv(x_t) for conv in self.convs]
        # 多尺度特征拼接
        fused = torch.cat(features, dim=1).permute(0,2,1)  # [bs, seq_len, dim]
        return self.fuse(fused)