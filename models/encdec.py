import math
import torch
import torch.nn as nn
from models.resnet import Resnet1D
from models.vq_model_dual import Normalize, nonlinearity
from models.resnet import RepeatFirstElementPad1d
import torch.nn.functional as F

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

class Encoderv2(nn.Module):
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
                 causal=False):
        super().__init__()
        
        self.causal = causal
        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        if causal:
            padding = RepeatFirstElementPad1d(padding=2)
            conv1 = nn.Conv1d(input_emb_width, width, 3, 1, 0)
            conv2 = nn.Conv1d(width, output_emb_width, 3, 1, 0)
            padding1 = RepeatFirstElementPad1d(padding=2 * pad_t)
            conv3 = nn.Conv1d(width, width, filter_t, stride_t, 0)
        else:
            padding = nn.Identity()
            conv1 = nn.Conv1d(input_emb_width, width, 3, 1, 1)
            conv2 = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            padding1 = nn.Identity()
            conv3 = nn.Conv1d(width, width, filter_t, stride_t, pad_t)
        blocks.append(padding)
        blocks.append(conv1)
        blocks.append(nn.ReLU())

        for i in range(down_t):
            # input_dim = width
            block = nn.Sequential(
                padding1,
                conv3,
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
            )
            blocks.append(block)
        blocks.append(padding)
        blocks.append(conv2)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder_cnn(nn.Module):
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
                 causal=False):
        super().__init__()
        
        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        # filter_t, pad_t = stride_t * 2, stride_t // 2
        if causal:
            # blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3的因果填充
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            if causal:
                causal_pad = (filter_t-1)
                block = nn.Sequential(
                    nn.ConstantPad1d((causal_pad,0), 0),  # 左侧填充
                    nn.Conv1d(input_dim, width, filter_t, stride_t, 0),
                    Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
                )
            else:
                block = nn.Sequential(
                    nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
                )
            blocks.append(block)
        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, motion_mask = None):
        if motion_mask is not None:
            x = x * motion_mask.unsqueeze(1)
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
    
class Decoder_wo_upsamplev2(nn.Module):
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
                 with_attn=False,
                 causal=False):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.with_attn = with_attn
        if causal:
            padding = RepeatFirstElementPad1d(padding=2)
            conv1 = nn.Conv1d(output_emb_width, width, 3, 1, 0)
            conv2 = nn.Conv1d(width, width, 3, 1, 0)
            conv3 = nn.Conv1d(width, input_emb_width, 3, 1, 0)
        else:
            padding = nn.Identity()
            conv1 = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            conv2 = nn.Conv1d(width, width, 3, 1, 1)
            conv3 = nn.Conv1d(width, input_emb_width, 3, 1, 1)
        blocks.append(padding)
        blocks.append(conv1)
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            if self.with_attn:
                block = nn.Sequential(
                    MotionAttention(width, num_heads=8),
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                    padding,
                    conv2,
                )
            else:
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                    padding,
                    conv2,
                )
            blocks.append(block)
        blocks.append(padding)
        blocks.append(conv2)
        blocks.append(nn.ReLU())
        blocks.append(padding)
        blocks.append(conv3)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)    

class Decoder_cnn(nn.Module):
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
                 causal=False):
        super().__init__()
        blocks = []
        self.causal = causal
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if causal:
            # blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3的因果填充
            blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            if causal:
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConstantPad1d((2,0), 0),
                    nn.Conv1d(width, out_dim, 3, 1, 0)
                )
            else:  
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(width, out_dim, 3, 1, 1)
                )
            blocks.append(block)
        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(width, width, 3, 1, 0))
            blocks.append(nn.ReLU())
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())
            blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder_wo_upsamplev1(nn.Module):
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
        self.with_attn = with_attn
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            if self.with_attn:
                block = nn.Sequential(
                    MotionAttention(width, num_heads=8),
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    nn.Conv1d(width, out_dim, 3, 1, 1)
                )
            else:
                block = nn.Sequential(
                    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                    nn.Conv1d(width, out_dim, 3, 1, 1)
                )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

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
    

class _CausalPadAndConv1d(nn.Module):
    """
    Helper module to handle causal or standard 'same' padding for nn.Conv1d.
    Mirrors the padding logic used in the provided Encoder_cnn.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, causal=False, dilation=1, groups=1, bias=True):
        super().__init__()
        self.causal = causal
        if self.causal:
            # Causal padding: (kernel_size - 1) for stride 1. For stride > 1, it's kernel_size - stride.
            if stride == 1:
                pad_amount = (kernel_size - 1) * dilation
            else: # stride > 1, specific to how Encoder_cnn handles downsampling
                pad_amount = kernel_size - stride # e.g., kernel=stride*2, pad_left = stride
            
            self.padding_layer = nn.ConstantPad1d((pad_amount, 0), 0)
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                                  padding=0, dilation=dilation, groups=groups, bias=bias)
        else:
            # Non-causal 'same' padding
            # For kernel=3, stride=1 => padding=1
            # For kernel=stride*2, stride>1 => padding=stride//2
            if kernel_size == 3 and stride == 1:
                conv_padding = 1
            elif stride > 1 and kernel_size == stride * 2:
                conv_padding = stride // 2
            else: # General fallback, try to make it 'same' for odd kernels
                conv_padding = (kernel_size - 1) // 2 if kernel_size % 2 == 1 else kernel_size // 2 -1

            self.padding_layer = nn.Identity()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=conv_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.padding_layer(x)
        return self.conv(x)

class MultiPartEncoder(nn.Module):
    def __init__(self,
                 parts_input_dims_dict: dict,     # e.g., {'Root': 7, 'R_Leg': 50, ...}
                 common_hidden_dim: int,          # Internal width for processing
                 parts_output_dims_dict: dict,    # e.g., {'Root': 64, 'R_Leg': 64, ...}
                 down_t: int,                     # Number of downsampling stages
                 stride_t: int,                   # Stride for temporal downsampling in each stage
                 depth: int,                      # Depth for Resnet1D blocks
                 dilation_growth_rate: int,       # Dilation growth rate for Resnet1D
                 activation: str = 'relu',
                 norm: str = None,
                 causal: bool = False,
                 enable_interaction: bool = True):
        super().__init__()

        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm'] # Fixed order
        self.num_parts = len(self.parts_name)
        self.causal = causal
        self.enable_interaction = enable_interaction

        # 1. Initial convolutions for each part
        self.initial_convs = nn.ModuleList()
        for part_name in self.parts_name:
            raw_dim = parts_input_dims_dict[part_name]
            self.initial_convs.append(
                _CausalPadAndConv1d(raw_dim, common_hidden_dim, kernel_size=3, stride=1, causal=self.causal)
            )
        
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            # Fallback or raise error
            self.activation_fn = nn.ReLU()
            print(f"Warning: Unsupported activation '{activation}', defaulting to ReLU.")

        # 2. Downsampling stages with per-part processing and inter-part interaction
        self.downsampling_stages = nn.ModuleList()
        current_h_dim = common_hidden_dim # The dimension of features for parts
        for _ in range(down_t):
            part_resnets_stage = nn.ModuleList()
            part_down_convs_stage = nn.ModuleList()
            
            for i_part in range(self.num_parts):
                part_resnets_stage.append(
                    Resnet1D(current_h_dim, depth, dilation_growth_rate,
                             activation=activation, norm=norm, causal=self.causal)
                )
                
                down_conv_kernel_size = stride_t * 2
                part_down_convs_stage.append(
                    _CausalPadAndConv1d(current_h_dim, current_h_dim, 
                                       kernel_size=down_conv_kernel_size,
                                       stride=stride_t, causal=self.causal)
                )
            
            # Interaction ResNet: operates on 'current_h_dim' channels, across 'num_parts' sequence
            # This interaction is NOT causal in the 'parts' dimension.
            interaction_resnet_stage = Resnet1D(current_h_dim, depth, dilation_growth_rate,
                                                activation=activation, norm=norm, causal=False) 

            self.downsampling_stages.append(nn.ModuleDict({
                'part_resnets': part_resnets_stage,
                'part_down_convs': part_down_convs_stage,
                'interaction_resnet': interaction_resnet_stage
            }))

        # 3. Final convolutions for each part
        self.final_convs = nn.ModuleList()
        for part_name in self.parts_name:
            part_output_dim = parts_output_dims_dict[part_name]
            self.final_convs.append(
                _CausalPadAndConv1d(current_h_dim, part_output_dim, kernel_size=3, stride=1, causal=self.causal)
            )

    def forward(self, parts_data_list: list):
        # Ensure parts_data_list has 6 tensors
        if not isinstance(parts_data_list, list) or len(parts_data_list) != self.num_parts:
            raise ValueError(f"Input must be a list of {self.num_parts} tensors.")

        # Initial projection and activation
        current_part_features = [
            self.initial_convs[i](parts_data_list[i]) for i in range(self.num_parts)
        ]
        current_part_features = [self.activation_fn(feat) for feat in current_part_features]

        # Downsampling and Interaction Loop
        for stage_module in self.downsampling_stages:
            # Per-part processing (ResNet + Downsampling Conv)
            processed_in_stage = []
            for i in range(self.num_parts):
                x = current_part_features[i]
                x = stage_module['part_resnets'][i](x)
                x = stage_module['part_down_convs'][i](x)
                processed_in_stage.append(x)
            
            if self.enable_interaction:
                # Interaction step
                # Features in processed_in_stage are (B, common_hidden_dim, T_current)
                # Stack to (B, common_hidden_dim, num_parts, T_current)
                try:
                    stacked_for_interaction = torch.stack(processed_in_stage, dim=2)
                except RuntimeError as e:
                    # This might happen if T_current is not the same for all parts after downsampling,
                    # which shouldn't occur if padding/stride is handled correctly.
                    print(f"Error stacking features for interaction. Shapes: {[f.shape for f in processed_in_stage]}")
                    raise e
                
                B, H, NumP, T_curr = stacked_for_interaction.shape
                
                # Reshape for ResNet1D: (Batch_eff, Channels, Seq_len_eff)
                # Batch_eff = B * T_curr, Channels = H (common_hidden_dim), Seq_len_eff = NumP
                to_interact_permuted = stacked_for_interaction.permute(0, 3, 1, 2).contiguous() # (B, T_curr, H, NumP)
                to_interact_reshaped = to_interact_permuted.view(B * T_curr, H, NumP)
                
                interacted_features = stage_module['interaction_resnet'](to_interact_reshaped) # (B*T_curr, H, NumP)
                
                interacted_unreshaped = interacted_features.view(B, T_curr, H, NumP)
                # Permute back to (B, H, NumP, T_curr)
                interacted_permuted_back = interacted_unreshaped.permute(0, 2, 3, 1).contiguous() 
                
                # Unstack into a list of part features (B, H, T_curr)
                current_part_features = list(torch.unbind(interacted_permuted_back, dim=2))
            else:
                # If interaction is disabled, use the per-part processed features directly
                current_part_features = processed_in_stage

        # Final projection
        output_part_features = [
            self.final_convs[i](current_part_features[i]) for i in range(self.num_parts)
        ]
            
        return output_part_features

class MultiPartEncoder_down2(nn.Module):
    def __init__(self,
                 parts_input_dims_dict: dict,     # e.g., {'Root': 7, 'R_Leg': 50, ...}
                 common_hidden_dim: int,          # Internal width for processing
                 parts_output_dims_dict: dict,    # e.g., {'Root': 64, 'R_Leg': 64, ...}
                 down_t: int,                     # Number of downsampling stages
                 stride_t: int,                   # Stride for temporal downsampling in each stage
                 depth: int,                      # Depth for Resnet1D blocks
                 dilation_growth_rate: int,       # Dilation growth rate for Resnet1D
                 activation: str = 'relu',
                 norm: str = None,
                 causal: bool = False,
                 enable_interaction: bool = True):
        super().__init__()

        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm'] # Fixed order
        self.num_parts = len(self.parts_name)
        self.causal = causal
        self.enable_interaction = enable_interaction

        # 1. Initial convolutions for each part
        self.initial_convs = nn.ModuleList()
        for part_name in self.parts_name:
            raw_dim = parts_input_dims_dict[part_name]
            self.initial_convs.append(
                _CausalPadAndConv1d(raw_dim, common_hidden_dim, kernel_size=3, stride=1, causal=self.causal)
            )
        
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            # Fallback or raise error
            self.activation_fn = nn.ReLU()
            print(f"Warning: Unsupported activation '{activation}', defaulting to ReLU.")

        # 2. Downsampling stages with per-part processing and inter-part interaction
        self.downsampling_stages = nn.ModuleList()
        current_h_dim = common_hidden_dim # The dimension of features for parts
        for i in range(2):
            part_resnets_stage = nn.ModuleList()
            part_down_convs_stage = nn.ModuleList()
            
            for i_part in range(self.num_parts):
                part_resnets_stage.append(
                    Resnet1D(current_h_dim, depth, dilation_growth_rate,
                             activation=activation, norm=norm, causal=self.causal)
                )
                if i == 0:
                    down_conv_kernel_size = stride_t * 2
                    part_down_convs_stage.append(
                        _CausalPadAndConv1d(current_h_dim, current_h_dim, 
                                        kernel_size=down_conv_kernel_size,
                                        stride=stride_t, causal=self.causal)
                    )
                else:
                    part_down_convs_stage.append(
                        nn.Identity()
                    )
            # Interaction ResNet: operates on 'current_h_dim' channels, across 'num_parts' sequence
            # This interaction is NOT causal in the 'parts' dimension.
            interaction_resnet_stage = Resnet1D(current_h_dim, depth, dilation_growth_rate,
                                                activation=activation, norm=norm, causal=False) 

            self.downsampling_stages.append(nn.ModuleDict({
                'part_resnets': part_resnets_stage,
                'part_down_convs': part_down_convs_stage,
                'interaction_resnet': interaction_resnet_stage
            }))

        # 3. Final convolutions for each part
        self.final_convs = nn.ModuleList()
        for part_name in self.parts_name:
            part_output_dim = parts_output_dims_dict[part_name]
            self.final_convs.append(
                _CausalPadAndConv1d(current_h_dim, part_output_dim, kernel_size=3, stride=1, causal=self.causal)
            )

    def forward(self, parts_data_list: list):
        # Ensure parts_data_list has 6 tensors
        if not isinstance(parts_data_list, list) or len(parts_data_list) != self.num_parts:
            raise ValueError(f"Input must be a list of {self.num_parts} tensors.")

        # Initial projection and activation
        current_part_features = [
            self.initial_convs[i](parts_data_list[i]) for i in range(self.num_parts)
        ]
        current_part_features = [self.activation_fn(feat) for feat in current_part_features]

        # Downsampling and Interaction Loop
        for stage_module in self.downsampling_stages:
            # Per-part processing (ResNet + Downsampling Conv)
            processed_in_stage = []
            for i in range(self.num_parts):
                x = current_part_features[i]
                x = stage_module['part_resnets'][i](x)
                x = stage_module['part_down_convs'][i](x)
                processed_in_stage.append(x)
            
            if self.enable_interaction:
                # Interaction step
                # Features in processed_in_stage are (B, common_hidden_dim, T_current)
                # Stack to (B, common_hidden_dim, num_parts, T_current)
                try:
                    stacked_for_interaction = torch.stack(processed_in_stage, dim=2)
                except RuntimeError as e:
                    # This might happen if T_current is not the same for all parts after downsampling,
                    # which shouldn't occur if padding/stride is handled correctly.
                    print(f"Error stacking features for interaction. Shapes: {[f.shape for f in processed_in_stage]}")
                    raise e
                
                B, H, NumP, T_curr = stacked_for_interaction.shape
                
                # Reshape for ResNet1D: (Batch_eff, Channels, Seq_len_eff)
                # Batch_eff = B * T_curr, Channels = H (common_hidden_dim), Seq_len_eff = NumP
                to_interact_permuted = stacked_for_interaction.permute(0, 3, 1, 2).contiguous() # (B, T_curr, H, NumP)
                to_interact_reshaped = to_interact_permuted.view(B * T_curr, H, NumP)
                
                interacted_features = stage_module['interaction_resnet'](to_interact_reshaped) # (B*T_curr, H, NumP)
                
                interacted_unreshaped = interacted_features.view(B, T_curr, H, NumP)
                # Permute back to (B, H, NumP, T_curr)
                interacted_permuted_back = interacted_unreshaped.permute(0, 2, 3, 1).contiguous() 
                
                # Unstack into a list of part features (B, H, T_curr)
                current_part_features = list(torch.unbind(interacted_permuted_back, dim=2))
            else:
                # If interaction is disabled, use the per-part processed features directly
                current_part_features = processed_in_stage

        # Final projection
        output_part_features = [
            self.final_convs[i](current_part_features[i]) for i in range(self.num_parts)
        ]
            
        return output_part_features