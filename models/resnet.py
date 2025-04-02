import torch.nn as nn
import torch
from encdec import RepeatFirstElementPad1d

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None, causal=False):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

    #     self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
    #     self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     
        # 因果卷积的特殊处理
        if causal:
            # 因果卷积需要左侧填充
            self.pad = RepeatFirstElementPad1d(padding=(dilation * 2, 0))  # 左侧填充2*dilation
            self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, 0, dilation)  # padding=0
        else:
            # 非因果卷积使用对称填充
            self.pad = nn.Identity()
            self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, dilation, dilation)
        # 第二层卷积（1x1卷积，不需要考虑因果性）
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
        
        x = self.pad(x)
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, causal=False):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, causal=causal) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
    
class RepeatFirstElementPad1d(nn.Module):
    """自定义填充层：用第一个元素重复填充左侧"""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        # x形状: [B, C, T]
        if self.padding == 0:
            return x
        # 取第一个元素并重复填充到左侧
        first_elem = x[:, :, :1]  # [B, C, 1]
        pad = first_elem.repeat(1, 1, self.padding)  # [B, C, padding]
        return torch.cat([pad, x], dim=2)  # [B, C, padding + T]  