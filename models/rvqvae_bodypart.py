import math
from einops import rearrange
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder, EnhancedDecoder, Decoder_wo_upsample, PureMotionDecoder
import torch.nn.functional as F
from models.lgvq import LGVQ, CausalTransformerEncoder, ContrastiveLossWithSTS, ContrastiveLossWithSTSV2, Dualsem_encoder, LGVQv2, LGVQv3, LGVQv4, LGVQv5
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.residual_vq import ResidualVQ
# from transformers import CLIPTextModel, CLIPTokenizer  # 使用Hugging Face版本

from transformers import AutoTokenizer, AutoModel
from transformers import CLIPTextModel, CLIPTokenizer
import os
from models.vqvae_bodypart import VQVAE_bodypart as VQVAE_ori
from models.layers.transformer import SpatialTemporalBlock
from models.resnet import Resnet1D
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 必须放在其他导入之前

class VQVAE_bodypart(nn.Module):
    def __init__(self,
                 args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        self.parts_code_nb = parts_code_nb
        self.parts_code_dim = parts_code_dim
        self.parts_output_dim = parts_output_dim
        self.parts_hidden_dim = parts_hidden_dim
        self.quantizer_type = args.quantizer
        self.down_t = down_t
        self.stride_t = stride_t
        self.depth = depth
        self.dilation_growth_rate = dilation_growth_rate
        self.activation = activation
        self.norm = norm

        if args.dataname == 't2m':
            parts_input_dim = {
                'Root': 7,
                'R_Leg': 50,
                'L_Leg': 50,
                'Backbone': 60,
                'R_Arm': 60,
                'L_Arm': 60,
            }
            self.parts_input_dim = parts_input_dim
            for name in self.parts_name:
                raw_dim = parts_input_dim[name]
                hidden_dim = parts_hidden_dim[name]
                output_dim = parts_output_dim[name]

                encoder = Encoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                decoder = Decoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                setattr(self, f'enc_{name}', encoder)
                setattr(self, f'dec_{name}', decoder)

                code_dim = parts_code_dim[name]
                # [Warning] code_dim (used in quantizer) must match the output_emb_width
                assert code_dim == output_dim
                nb_code = parts_code_nb[name]
                rvqvae_config = {
                    'num_quantizers': args.num_quantizers,
                    'shared_codebook': args.shared_codebook,
                    'quantize_dropout_prob': args.quantize_dropout_prob,
                    'quantize_dropout_cutoff_index': 0,
                    'nb_code': nb_code,
                    'code_dim': code_dim,
                    'args': args,
                }
                quantizer = ResidualVQ(**rvqvae_config)
                setattr(self, f'quantizer_{name}', quantizer)

        elif args.dataname == 'kit':
            parts_input_dim = {
                'Root': 7,
                'R_Leg': 62,
                'L_Leg': 62,
                'Backbone': 48,
                'R_Arm': 48,
                'L_Arm': 48,
            }
            self.parts_input_dim = parts_input_dim
            for name in self.parts_name:
                raw_dim = parts_input_dim[name]
                hidden_dim = parts_hidden_dim[name]
                output_dim = parts_output_dim[name]

                encoder = Encoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                decoder = Decoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                setattr(self, f'enc_{name}', encoder)
                setattr(self, f'dec_{name}', decoder)

                code_dim = parts_code_dim[name]
                # [Warning] code_dim (used in quantizer) must match the output_emb_width
                assert code_dim == output_dim
                nb_code = parts_code_nb[name]
                rvqvae_config = {
                    'num_quantizers': args.num_quantizers,
                    'shared_codebook': args.shared_codebook,
                    'quantize_dropout_prob': args.quantize_dropout_prob,
                    'quantize_dropout_cutoff_index': 0,
                    'nb_code': nb_code,
                    'code_dim': code_dim,
                    'args': args,
                }
                quantizer = ResidualVQ(**rvqvae_config)
                # if args.quantizer == "ema_reset":
                #     quantizer = QuantizeEMAReset(nb_code, code_dim, args)
                # elif args.quantizer == "orig":
                #     quantizer = Quantizer(nb_code, code_dim, 1.0)
                # elif args.quantizer == "ema":
                #     quantizer = QuantizeEMA(nb_code, code_dim, args)
                # elif args.quantizer == "reset":
                #     quantizer = QuantizeReset(nb_code, code_dim, args)
                setattr(self, f'quantizer_{name}', quantizer)
        
        else:
            raise Exception()


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, parts, motion_mask = None):
        """
        This is used in training transformer (train_t2m_trans.py and the parts ver.),
          for getting the embedding(also named tokens, discrete repre) of motions.

        parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, nframes, 7)
          R_Leg:    (B, nframes, 50)
          L_Leg:    (B, nframes, 50)
          Backbone: (B, nframes, 60)
          R_Arm:    (B, nframes, 60)
          L_Arm:    (B, nframes, 60)
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        tokenized_parts = []
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            N, T, _ = x.shape
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder = encoder(x_in)  # (B, out_dim, nframes)
            x_encoder = self.postprocess(x_encoder)  # (B, nframes, out_dim)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (B*nframes, out_dim)

            # Quantization
            quantizer = getattr(self, f'quantizer_{name}')
            code_idx = quantizer.quantize(x_encoder)  # (B*nframes, out_dim) --> (B*nframes)
            code_idx = code_idx.view(N, -1)  # (B, nframes)
            if motion_mask is not None:
                code_idx[~motion_mask.bool()] = quantizer.nb_code

            tokenized_parts.append(code_idx)

        return tokenized_parts


    def forward(self, parts, caption=None):
        """
        Forwarding.
        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, nframes, 7)
          R_Leg:    (B, nframes, 50)
          L_Leg:    (B, nframes, 50)
          Backbone: (B, nframes, 60)
          R_Arm:    (B, nframes, 60)
          L_Arm:    (B, nframes, 60)
        :return:
        """

        # [Note] remember to be consistent with the self.parts_name when use the x.
        #   self.parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        x_out_list = []
        loss_list = []
        perplexity_list = []
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder = encoder(x_in)

            # Quantization
            quantizer = getattr(self, f'quantizer_{name}')
            x_quantized, code_idx, loss, perplexity = quantizer(x_encoder, sample_codebook_temp=0.5)

            # Decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_quantized)

            # Postprocess
            x_out = self.postprocess(x_decoder)  # (B, in_dim, nframes) ==> (B, nframes, in_dim)

            x_out_list.append(x_out)
            loss_list.append(loss)
            perplexity_list.append(perplexity)

        # Return the list of x_out, loss, perplexity
        return x_out_list, loss_list, perplexity_list


    def forward_decoder(self, parts):
        """
        This function will be used in evaluation of transformer (eval_bodypart.py).
          It is used to decode the predicted index motion from the transformer.

        Only support BatchSize == 1.

        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, codes_len)      B == 1
          R_Leg:    (B, codes_len)      B == 1
          L_Leg:    (B, codes_len)      B == 1
          Backbone: (B, codes_len)      B == 1
          R_Arm:    (B, codes_len)      B == 1
          L_Arm:    (B, codes_len_)     B == 1

          The input parts should have the same codes_len.
          If not, these parts codes should be truncated according to min codes_len before input into this function

        :return:
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        parts_out = []
        base_codes_len = parts[0].shape[1]
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            assert x.shape[0] == 1  # ensure batch size is 1
            codes_len = x.shape[1]
            assert codes_len == base_codes_len  # make sure all parts has same codes_len

            quantizer = getattr(self, f'quantizer_{name}')
            x_d = quantizer.dequantize(x)  # (B, codes_len) => (B, codes_len, code_dim), B == 1

            # It seems the .view() operation does not bring any change.
            #   The code probably is just adapted from the quantizer's code
            x_d = x_d.view(1, codes_len, -1).permute(0, 2, 1).contiguous()  # (B, code_dim, codes_len)

            # decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_d)  # (B, raw_motion_dim, seq_len)
            x_out = self.postprocess(x_decoder)  # (B, seq_len, raw_motion_dim)

            parts_out.append(x_out)

        return parts_out


    def forward_decoder_batch(self, parts):
        """
        Decode the quantized motion to raw motion

        Support computation in batch.

        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, codes_len)
          R_Leg:    (B, codes_len)
          L_Leg:    (B, codes_len)
          Backbone: (B, codes_len)
          R_Arm:    (B, codes_len)
          L_Arm:    (B, codes_len_)

          The input parts should have the same codes_len.
          If not, these parts codes should be truncated according to min codes_len before input into this function

        :return:
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        parts_out = []
        base_codes_len = parts[0].shape[1]
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            B = x.shape[0] # ensure batch size is 1
            codes_len = x.shape[1]
            assert codes_len == base_codes_len  # make sure all parts has same codes_len

            quantizer = getattr(self, f'quantizer_{name}')
            x_d = quantizer.dequantize(x)  # (B, codes_len) => (B, codes_len, code_dim), B == 1

            # It seems the .view() operation does not bring any change.
            #   The code probably is just adapted from the quantizer's code
            x_d = x_d.view(B, codes_len, -1).permute(0, 2, 1).contiguous()  # (B, code_dim, codes_len)

            # decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_d)  # (B, raw_motion_dim, seq_len)
            x_out = self.postprocess(x_decoder)  # (B, seq_len, raw_motion_dim)

            parts_out.append(x_out)

        return parts_out

class TemporalDownsampler(nn.Module):
    """时间维度1/4降采样模块"""
    def __init__(self, d_model):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//4, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, T//4, C]
        return self.norm(x)

class TemporalDownsamplerV2(nn.Module):
    """时间维度1/4降采样模块"""
    def __init__(self, d_model):
        super().__init__()
        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        self.conv_layers2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)
        )
        self.conv_layers3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//4, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x1 = self.conv_layers1(x)
        x2 = self.conv_layers2(x)
        x3 = self.conv_layers3(x)
        x = (x1 + x2 + x3)/3
        x = x.permute(0, 2, 1)  # [B, T//4, C]
        return self.norm(x)




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
def visualize_tsne(motion_feat, text_feat, filename='tsne_visualization.png'):
    """
    使用t-SNE对motion_feat和text_feat进行可视化，并保存到本地文件
    :param motion_feat: 动作特征 (torch.Tensor)
    :param text_feat: 文本特征 (torch.Tensor)
    :param filename: 保存的文件名
    """
    # 将特征转换为numpy数组
    motion_feat_np = motion_feat.cpu().detach().numpy()
    text_feat_np = text_feat.cpu().detach().numpy()
    
    # 合并特征
    combined_feat = np.concatenate((motion_feat_np, text_feat_np), axis=0)
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_feat)
    
    # 分离降维后的结果
    motion_tsne = tsne_results[:motion_feat_np.shape[0]]
    text_tsne = tsne_results[motion_feat_np.shape[0]:]
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(motion_tsne[:, 0], motion_tsne[:, 1], label='Motion Features', alpha=0.5)
    plt.scatter(text_tsne[:, 0], text_tsne[:, 1], label='Text Features', alpha=0.5)
    plt.legend()
    plt.title("t-SNE Visualization of Motion and Text Features")
    plt.savefig(filename)
    plt.close()

def visualize_similarity_matrix(sim_matrix, filename='similarity_matrix.png'):
    """
    可视化相似度矩阵并保存到本地文件
    :param sim_matrix: 相似度矩阵 (torch.Tensor)
    :param filename: 保存的文件名
    """
    sim_matrix = sim_matrix.cpu().detach().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title("相似度矩阵")
    plt.savefig(filename)
    plt.close()

class EnhancedPartFusion(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                #  num_layers=4,
                 use_zero_init=True):
        super().__init__()
        
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        self.time_position = nn.Parameter(torch.randn(1, 128, d_model))  # 时间步编码(假设最大序列长度128)
        
        # 时间下采样模块
        self.temporal_downsample = TemporalDownsampler(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, d_model//2, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model//2, d_model, 3, padding=1)
            ) for dim in part_dims
        ])
        
        # 空间Transformer配置
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            ),
            num_layers=2
        )
        # 时间Transformer配置
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead//2,
                dim_feedforward=2*d_model,
                batch_first=True
            ),
            num_layers=2
        )
        # 残差连接优化
        self.res_conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        if use_zero_init:
            nn.init.zeros_(self.res_conv.weight)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = self.temporal_downsample(rearrange(spatial_cube, 'b t p d -> (b p) t d'))
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B, p=6)
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 6, d]
        
        # 时间维度交互 (运动连续性)
        temporal_feat = spatial_cube + self.time_position[:, :T//4, :]
        temporal_feat = self.temporal_transformer(temporal_feat)  # [B, T//4, p*d]
        fused_feat = rearrange(temporal_feat, '(b p) t d -> b (p t) d', b=B, p=6) + rearrange(spatial_feat, '(b t) p d -> b (p t) d', b=B, p=6)
        
        # # 处理文本缺失情况
        # if text_feature is None:
        #     # 生成自适应占位符
        #     null_text = self.null_text_embed.expand(B, -1, -1)  # [B, 1, d_model]
        #     text_feature = null_text
        
        # # 文本存在性门控
        # gate = self.text_gate(text_feature.mean(dim=1))  # [B, 1]
        # # 跨模态相对注意力
        # text_key = text_feature * gate.unsqueeze(1)
        
        # # 跨模态注意力
        # attn_output, _ = self.cross_attn(
        #     query=fused_feat,
        #     key=text_key,
        #     value=text_feature,
        #     need_weights=False
        # )
        # fused_feat = attn_output
        
        # 残差连接增强
        return rearrange(self.res_conv(fused_feat.transpose(1,2)).transpose(1,2), 'b (p t) d -> b t p d', b=B, p=6), rearrange(fused_feat, 'b (p t) d -> b t p d', b=B, p=6)

class EnhancedPartFusionV2(nn.Module):
    def __init__(self, args,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                #  num_layers=4,
                 use_zero_init=True):
        super().__init__()
        
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        self.time_position = nn.Parameter(torch.randn(1, 128, d_model))  # 时间步编码(假设最大序列长度128)
        
        # 时间下采样模块
        self.temporal_downsample = TemporalDownsamplerV2(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, d_model//2, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model//2, d_model, 3, padding=1)
            ) for dim in part_dims
        ])
        
        # 空间Transformer配置
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            ),
            num_layers=2
        )
        # 时间Transformer配置
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead//2,
                dim_feedforward=2*d_model,
                batch_first=True
            ),
            num_layers=2
        )
        # 残差连接优化
        self.res_conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        if use_zero_init:
            nn.init.zeros_(self.res_conv.weight)
        
        # self.quantizer = QuantizeEMAReset(1024, d_model, args)
        rvqvae_config = {
                    'num_quantizers': args.num_quantizers_global,
                    'shared_codebook': args.shared_codebook,
                    'quantize_dropout_prob': args.quantize_dropout_prob,
                    'quantize_dropout_cutoff_index': 0,
                    'nb_code': 256,
                    'code_dim': d_model,
                    'args': args,
                }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = self.temporal_downsample(rearrange(spatial_cube, 'b t p d -> (b p) t d'))
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B, p=6)
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 6, d]
        
        # 时间维度交互 (运动连续性)
        temporal_feat = rearrange(spatial_feat, '(b t) p d -> (b p) t d', b=B, p=6)
        temporal_feat = temporal_feat + self.time_position[:, :T//4, :]
        temporal_feat = self.temporal_transformer(temporal_feat)  # [B, T//4, p*d]
        # fused_feat = rearrange(temporal_feat, '(b p) t d -> b (p t) d', b=B, p=6) + rearrange(spatial_feat, '(b t) p d -> b (p t) d', b=B, p=6)
        # fused_feat = temporal_feat
        fused_feat, code_idx, loss, perplexity = self.quantizer(rearrange(temporal_feat, '(b p) t d -> b d (p t)', b=B, p=6), sample_codebook_temp=0.5)
        
        # 残差连接增强
        return rearrange(fused_feat, 'b d (p t) -> b t p d', b=B, p=6), loss, perplexity

class StructuredAttention(nn.MultiheadAttention):
    """修正掩码方向和扩展方式的注意力层"""
    def __init__(self, embed_dim, num_heads):
        super().__init__(embed_dim, num_heads)
        self.register_buffer('base_mask', self._build_connection_matrix())
        
    def _build_connection_matrix(self):
        """构建符合拓扑约束的6x6连接矩阵（True表示需要屏蔽）"""
        mask = torch.zeros(6, 6, dtype=torch.bool)
        # 设置需要屏蔽的位置为True
        mask[1, [2,4,5]] = True  # R_Leg
        mask[2, [1,4,5]] = True  # L_Leg 
        mask[4, [1,2,5]] = True  # R_Arm
        mask[5, [1,2,4]] = True  # L_Arm
        return mask

    def forward(self, query, key, value, attn_mask=None, **kwargs):     
        # 确保使用float掩码类型
        return super().forward(
            query, key, value,
            attn_mask=self.base_mask,  # 转换为float类型掩码
            **kwargs
        )
        
class StructuredAttention_withtoken(nn.MultiheadAttention):
    """修正掩码方向和扩展方式的注意力层"""
    def __init__(self, embed_dim, num_heads):
        super().__init__(embed_dim, num_heads)
        self.register_buffer('base_mask', self._build_connection_matrix())
        
    def _build_connection_matrix(self):
        """构建符合拓扑约束的6x6连接矩阵（True表示需要屏蔽）"""
        mask = torch.zeros(7, 7, dtype=torch.bool)
        # 设置需要屏蔽的位置为True ['cls', 'Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        mask[2, [3,5,6]] = True  # R_Leg
        mask[3, [2,5,6]] = True  # L_Leg 
        mask[5, [2,3,6]] = True  # R_Arm
        mask[6, [2,3,5]] = True  # L_Arm
        return mask

    def forward(self, query, key, value, attn_mask=None, **kwargs):     
        # 确保使用float掩码类型
        return super().forward(
            query, key, value,
            attn_mask=self.base_mask,  # 转换为float类型掩码
            **kwargs
        )

class DynamicProjection(nn.Module):
    """动态特征投影"""
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Conv1d(in_dim, d_model//2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, 3, padding=1)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        base = self.base_proj(x)
        gate = self.gate(base)
        return base * gate

# 只做全身空间的交互
class EnhancedPartFusionV4(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 ):
        super().__init__()
        # 全身特征聚合Token（可学习参数）
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 时间下采样模块
        self.temporal_downsample = TemporalDownsamplerV2(d_model)
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 带结构约束的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        # 替换自注意力机制
        # encoder_layer.self_attn = StructuredAttention(d_model, nhead)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T // 4, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = self.temporal_downsample(rearrange(spatial_cube, 'b t p d -> (b p) t d'))
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> p (b t) d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        global_feat = spatial_feat[0, :, :]  # 取第一个位置的特征
        # 残差连接增强
        return rearrange(global_feat, '(b t) d -> b t d', b=B)

# 只做全身空间的交互, vit结构作为encoder
class EnhancedPartFusionV5(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 时间下采样模块
        self.temporal_downsample = TemporalDownsamplerV2(d_model)
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 带结构约束的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        # 替换自注意力机制
        # encoder_layer.self_attn = StructuredAttention_withtoken(d_model, nhead)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T // 4, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = self.temporal_downsample(rearrange(spatial_cube, 'b t p d -> (b p) t d'))
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> p (b t) d', b=B, p=7)
        global_feat = spatial_feat[0, :, :]  # 取第一个位置的特征
        # 残差连接增强
        return rearrange(global_feat, '(b t) d -> b t d', b=B), rearrange(spatial_feat[1:,...], 'p (b t) d -> p b t d', b=B)



# 全身空间的交互, 和时间的交互 vit结构作为encoder, 不做限制的transformer
class EnhancedPartFusionV6(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 时间下采样模块
        # self.temporal_downsample = TemporalDownsamplerV2(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature, motion_mask=None):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        # causal_mask = torch.triu(torch.ones(T, T, device=spatial_feat.device), diagonal=1).bool()
        # time_feat = self.time_transformer(time_feat)  # [T, B*7, d]
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [T, B*7, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)


# 只做全身空间的交互, vit结构作为encoder, 不做限制的transformer,相较于V5版本，只区别于V7没有下采样部分。
class EnhancedPartFusionV7(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        feature = rearrange(spatial_feat, '(b t) p d-> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

class GeneratorBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 空间注意力（处理 body_part 维度）
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # 时间注意力（处理 seq_len 维度）
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def _positional_encoding(self, position, d_model):
        # 正弦/余弦位置编码函数
        pe = torch.zeros(position.shape[0], d_model, device=position.device)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=position.device).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe

    def forward(self, x):
        """
        输入形状: [bs, body_part, seq_len, d_model]
        输出形状: [bs, body_part, seq_len, d_model]
        """
        bs, body_part, seq_len, d_model = x.shape
        

        # --- 空间注意力 ---
        # 合并 bs 和 seq_len，处理 body_part 维度
        spatial_input = x.permute(0, 2, 1, 3).reshape(bs*seq_len, body_part, d_model)
        attn_output_spatial, _ = self.spatial_attention(spatial_input, spatial_input, spatial_input)
        attn_output_spatial = attn_output_spatial.reshape(bs, seq_len, body_part, d_model).permute(0, 2, 1, 3)
        x = self.norm1(x + self.dropout(attn_output_spatial))  # Add & Norm

        # --- 时间注意力 ---
        # 合并 bs 和 body_part，处理 seq_len 维度
        temporal_input = x.permute(0, 1, 3, 2).reshape(bs*body_part, seq_len, d_model)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_output_temporal, _ = self.temporal_attention(temporal_input, temporal_input, temporal_input, attn_mask = attn_mask)
        attn_output_temporal = attn_output_temporal.reshape(bs, body_part, d_model, seq_len).permute(0, 1, 3, 2)
        x = self.norm2(x + self.dropout(attn_output_temporal))  # Add & Norm

        # --- MLP ---
        mlp_output = self.mlp(x)
        x = self.norm3(x + self.dropout(mlp_output))
        
        return x
    
# 不完全的实现，丢弃，使用V9版本
class EnhancedPartFusionV8(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # # 增强的位置编码体系
        # self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        self.spatial_temporal = nn.ModuleList([
            GeneratorBlock(d_model, nhead) for _ in range(num_layers)
        ])

    def _positional_encoding(self, position, d_model):
        # 正弦/余弦位置编码函数
        pe = torch.zeros(position.shape[0], d_model, device=position.device)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=position.device).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe
    
    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            part_embeds.append(proj_feat)
            # 部件类型编码
            # part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        spatial_feat = rearrange(fused_feat, '(b t) p d-> b p t d', b=B, p=7)
        # 引入时空位置编码
        bs, body_part, seq_len, d_model = spatial_feat.shape
        spatial_pos = torch.arange(body_part, device=spatial_feat.device).float()
        pe_space = self._positional_encoding(spatial_pos, d_model)  # [body_part, d_model]
        pe_space = pe_space.unsqueeze(0).unsqueeze(2)               # [1, body_part, 1, d_model]
        
        # 生成时间位置编码 (seq_len维度)
        temporal_pos = torch.arange(seq_len, device=spatial_feat.device).float()
        pe_time = self._positional_encoding(temporal_pos, d_model)  # [seq_len, d_model]
        pe_time = pe_time.unsqueeze(0).unsqueeze(1)                 # [1, 1, seq_len, d_model]
        
        # 将位置编码注入输入
        spatial_feat = spatial_feat + pe_space + pe_time  # [bs, body_part, seq_len, d_model]
        
        for block in self.spatial_temporal:
            spatial_feat = block(spatial_feat)
        feature = rearrange(spatial_feat, 'b p t d-> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

class EnhancedPartFusionV9(nn.Module):
    def __init__(self, causal=False,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        self.spatial_temporal = nn.ModuleList([
            SpatialTemporalBlock(d_model, nhead, causal=causal) for _ in range(num_layers)
        ])
    
    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            part_embeds.append(proj_feat)
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        spatial_feat = rearrange(fused_feat, '(b t) p d-> b p t d', b=B, p=7)
        
        for block in self.spatial_temporal:
            spatial_feat = block(spatial_feat)
        feature = rearrange(spatial_feat, 'b p t d-> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

class SpatialTemporalBlock(nn.Module):
    """自定义空间-时间注意力块"""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, causal=False):
        super().__init__()
        # 空间注意力组件
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 时间注意力组件（因果）
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # MLP组件
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.causal = causal
        # 激活函数
        self.activation = nn.GELU()

    def forward(self, src, spatial_mask=None, temporal_mask=None):
        # 空间注意力阶段
        B, T, P, D = src.shape  # [Batch, Time, Parts, Dim]
        
        # 展开空间维度
        spatial_src = rearrange(src, 'b t p d -> (b t) p d')  # [B*T, P, D]
        
        # 空间注意力（部件间交互）
        spatial_src2 = self.spatial_attn(
            spatial_src, spatial_src, spatial_src,
            key_padding_mask=spatial_mask
        )[0]
        spatial_src = spatial_src + self.dropout1(spatial_src2)
        spatial_src = self.norm1(spatial_src)
        
        # 重组回时空结构
        src = rearrange(spatial_src, '(b t) p d -> b t p d', b=B, t=T)
        
        # 时间注意力阶段（因果）
        temporal_src = rearrange(src, 'b t p d -> (b p) t d')  # [B*P, T, D]
        
        # 生成因果掩码
        if temporal_mask is None and self.causal == True:
            L = temporal_src.size(1)
            temporal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=src.device),
                diagonal=1
            )
        
        # 时间注意力（带因果约束）
        temporal_src2 = self.temporal_attn(
            temporal_src, temporal_src, temporal_src,
            attn_mask=temporal_mask
        )[0]
        temporal_src = temporal_src + self.dropout2(temporal_src2)
        temporal_src = self.norm2(temporal_src)
        
        # 重组回最终维度
        src = rearrange(temporal_src, '(b p) t d -> b t p d', b=B, p=P)
        
        # MLP阶段
        mlp_src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(mlp_src)
        src = self.norm3(src)
        
        return src

class EnhancedPartFusionV10(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 max_time_steps=196):
        super().__init__()
        # 初始化全局token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 增强的位置编码系统
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        self.time_position = nn.Parameter(torch.randn(1, max_time_steps, d_model))  # 可学习时间编码
        
        # 动态特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        
        # 构建自定义Transformer块
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model
            ) for _ in range(num_layers)
        ])
        
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 位置编码的特殊初始化
        nn.init.normal_(self.time_position, mean=0, std=0.02)
        nn.init.normal_(self.global_token, mean=0, std=0.02)

    def forward(self, parts_feature):
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 部件特征预处理
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 添加部件位置编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
        
        # 构建时空立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        
        # 添加全局token并扩展时间维度
        global_tokens = self.global_token.expand(B, T, -1, -1)  # [B, T, 1, d_model]
        fused_feat = torch.cat([global_tokens, spatial_cube], dim=2)  # [B, T, 7, d_model]
        
        # 注入时间位置编码
        fused_feat += self.time_position[:, :T, None, :]  # 广播到部件维度
        
        # 通过多个空间-时间块
        for block in self.blocks:
            fused_feat = block(fused_feat)
        
        # 分解输出特征
        global_feat = fused_feat[:, :, 0, :]  # [B, T, d_model]
        part_feats = fused_feat[:, :, 1:, :]  # [B, T, 6, d_model]
        
        return global_feat, rearrange(part_feats, 'b t p d -> p b t d')

# 全身空间的交互, 和时间的交互 vit结构作为encoder, 不做限制的transformer, 在V6版本基础上加入时间位置编码
class EnhancedPartFusionV12(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 max_time_steps = 196,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        self.time_position = nn.Parameter(torch.randn(1, max_time_steps, d_model))  # 可学习时间编码s
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> b t p d', b=B, p=7)
        # 注入时间位置编码
        time_feat += self.time_position[:, :T, None, :]  # 广播到部件维度
        time_feat = rearrange(time_feat, 'b t p d-> (b p) t d', b=B, p=7)
        time_feat = self.time_transformer(time_feat)  # [T, B*7, d]
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

class TemporalDownsamplerV3(nn.Module):
    """时间维度1/4降采样模块"""
    def __init__(self, d_model):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        # self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//4, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, T//4, C]
        return x
    
class TemporalDownsamplerV4(nn.Module):
    """时间维度1/4降采样模块"""
    def __init__(self, d_model,
                 down_t = 2,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(d_model, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, d_model, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=500):
        """
        Args:
            channels (int): 输入特征的通道数（需与下采样输出通道一致）
            max_len (int): 支持的最大时间步长
        """
        super().__init__()
        
        # 创建位置编码矩阵 (可扩展的缓冲区)
        position = torch.arange(max_len).unsqueeze(1)                    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, channels, 2).float() * 
            (-math.log(10000.0) / channels)
        )                                                                 # (channels//2,)
        
        pe = torch.zeros(max_len, channels)                              # (max_len, channels)
        pe[:, 0::2] = torch.sin(position * div_term)                     # 偶数列使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)                     # 奇数列使用余弦
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1, 2))       # (1, channels, max_len)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 (batch_size, channels, time_steps)
        Returns:
            输出张量，形状与输入相同
        """
        x = x + self.pe[:, :, :x.size(2)]   # 动态截取所需时间步
        return x

# 全身空间的交互, 和时间的交互 vit结构作为encoder, 不做限制的transformer, 在V6版本基础上加入时间位置编码
class EnhancedPartFusionV13(nn.Module):
    def __init__(self,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 causal=False,
                 position=0,
                 max_time_steps = 196,
                 ):
        super().__init__()
        self.position = position
        self.d_model = d_model
        self.part_dims = part_dims
        if position == 0:
            pass
        elif position == 1:
            self.time_position = nn.Parameter(torch.randn(1, max_time_steps, d_model))  # 可学习时间编码
        elif position == 2:
            self.time_position = PositionalEncoding(d_model)
        else:
            raise ValueError('position参数错误')
        # self.time_downsampler = TemporalDownsamplerV4(d_model)
        self.time_downsampler = TemporalDownsamplerV2(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        if causal:
            self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)
        else:
            self.time_transformer = nn.TransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            part_embeds.append(proj_feat)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        time_feat = torch.stack(part_embeds, dim=2)
        if torch.any(torch.isnan(time_feat)):
            print('nan in time_feat')
        time_feat = rearrange(time_feat, 'b t p d -> (b p) t d')
        time_feat = self.time_downsampler(time_feat)
        if self.position == 1:
            # time_feat = rearrange(time_feat, '(b p) t d -> b t p d')
            time_feat += self.time_position[:, :T//4, :]
        elif self.position == 2:
            time_feat = self.time_position(time_feat.permute(0, 2, 1)).permute(0, 2, 1)
        time_feat = self.time_transformer(time_feat)  # [B*7, T, d]
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=6)
        if torch.any(torch.isnan(feature)):
            print('nan in feature')
        return feature, rearrange(feature, 'b t p d -> p b t d', b=B)

class EnhancedPartFusionV14(nn.Module):
    def __init__(self,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 causal=False,
                 position=0,
                 max_time_steps = 196,
                 ):
        super().__init__()
        self.position = position
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        if position == 0:
            pass
        elif position == 1:
            self.time_position = nn.Parameter(torch.randn(1, max_time_steps, d_model))  # 可学习时间编码
        elif position == 2:
            self.time_position = PositionalEncoding(d_model)
        else:
            raise ValueError('position参数错误')
        self.time_downsampler = TemporalDownsamplerV3(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if causal:
            self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)
        else:
            self.time_transformer = nn.TransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            part_embeds.append(proj_feat)

        # 构建时空特征立方体 [B, T, 6, d_model]
        time_feat = torch.stack(part_embeds, dim=2)
        time_feat = rearrange(time_feat, 'b t p d -> (b p) t d')
        time_feat = self.time_downsampler(time_feat)
        if self.position == 1:
            # time_feat = rearrange(time_feat, '(b p) t d -> b t p d')
            time_feat += self.time_position[:, :T//4, :]
        elif self.position == 2:
            time_feat = self.time_position(time_feat.permute(0, 2, 1)).permute(0, 2, 1)
        time_feat = self.time_transformer(time_feat)  # [B*7, T, d]
        
        spatial_cube = rearrange(time_feat, '(b p) t d -> (b t) p d', b=B, p=6)
        # global_tokens = self.global_token.expand(B*T, -1, -1)
        spatial_cube = spatial_cube + self.part_position.weight[None, :, :]
        spatial_feature = self.spatial_transformer(spatial_cube)
        
        feature = rearrange(spatial_feature, '(b t) p d -> b t p d', b=B, p=6)
        return feature, rearrange(feature, 'b t p d -> p b t d', b=B)

class DynamicProjectionV2(nn.Module):
    """动态特征投影"""
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Conv1d(in_dim, d_model//2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.LayerNorm([d_model, 1]),  # 新增 LayerNorm
            nn.Conv1d(d_model, d_model, 1),
            nn.Sigmoid()
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        base = self.base_proj(x)
        gate = self.gate(base)
        return base * gate


# 全身空间的交互, 和时间的交互 vit结构作为encoder, 不做限制的transformer, 在V6版本基础上加入时间位置编码
class EnhancedPartFusionV15(EnhancedPartFusionV13):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjectionV2(dim, self.d_model) for dim in self.part_dims
        ])

    def forward(self, parts_feature):
        return super().forward(parts_feature)

# 全身空间的交互, 和时间的交互 vit结构作为encoder, 不做限制的transformer,与V6的区别是加了时间下采样
class EnhancedPartFusionV16(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 position=1,
                 causal=True):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 时间下采样模块
        self.temporal_downsample = TemporalDownsamplerV2(d_model)
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T // 4, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = self.temporal_downsample(rearrange(spatial_cube, 'b t p d -> (b p) t d'))
        # spatial_cube = rearrange(spatial_cube, '(b p) t d -> (b p) t d',b=B)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        time_feat = self.time_transformer(time_feat)  # [T, B*7, d]
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)


# 在V6版本上加入全局特征学习
class EnhancedPartFusionV17(nn.Module):
    def __init__(self,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 causal=False,
                 position=0,
                 max_time_steps = 196,
                 ):
        super().__init__()
        self.position = position
        self.d_model = d_model
        self.part_dims = part_dims
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        
        if position == 0:
            pass
        elif position == 1:
            self.time_position = nn.Parameter(torch.randn(1, max_time_steps, d_model))  # 可学习时间编码
        elif position == 2:
            self.time_position = PositionalEncoding(d_model)
        else:
            raise ValueError('position参数错误')
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if causal:
            self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)
        else:
            self.time_transformer = nn.TransformerEncoder(time_encoder_layer, num_layers=num_layers)
        
    def forward(self, parts_feature):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            part_embeds.append(proj_feat)
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间交互
        fused_feat = self.spatial_transformer(fused_feat)
        time_feat = rearrange(fused_feat, '(b t) p d -> (b p) t d', b=B)
        if self.position == 1:
            time_feat += self.time_position[:, :T, :]
        elif self.position == 2:
            time_feat = self.time_position(time_feat.permute(0, 2, 1)).permute(0, 2, 1)
        time_feat = self.time_transformer(time_feat)  # [B*7, T, d]
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feature = feature[:, :, 0, :]  # 取第一个位置的特征
        
        return global_feature, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

# V6版本去除全局特征
class EnhancedPartFusionV20(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature, motion_mask=None):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(spatial_cube, '(b p) t d-> (b t) p d', b=B, p=6)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=6)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(6, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [T, B*7, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=6)
        # 残差连接增强
        return rearrange(feature, 'b t p d -> p b t d', b=B)


# 全身空间的交互
class EnhancedPartFusionV5p(nn.Module):
    def __init__(self, 
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        # 层级式特征投影
        self.part_projs = nn.ModuleList([
            DynamicProjection(dim, d_model) for dim in part_dims
        ])
        # 不带结构约束的空间Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, parts_feature, motion_mask=None):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, (proj, feat) in enumerate(zip(self.part_projs, parts_feature)):
            # 时间维度卷积处理
            feat = feat.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            proj_feat = proj(feat.float()).permute(0, 2, 1)  # [B, T, d_model]
            
            # 部件类型编码
            part_embeds.append(proj_feat + self.part_position.weight[i][None, None, :])
            
        global_tokens = self.global_token.expand(B*T, -1, -1)
        
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_tokens,
            rearrange(spatial_cube, '(b p) t d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        feature = rearrange(spatial_feat, '(b t) p d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :]  # 取第一个位置的特征
        # 残差连接增强
        return global_feat, rearrange(feature[:, :, 1:, :], 'b t p d -> p b t d', b=B)

class EnhancedVQVAE(nn.Module):
    def __init__(self, args,
                 original_vqvae: VQVAE_bodypart,
                 text_encoder,
                 parts_output_dim,
                 d_model=256,
                 contrastive_temp=0.07):
        super().__init__()
        # 原始组件
        self.original_vqvae = original_vqvae
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusion(d_model=self.d_model, part_dims=part_dims)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)
        # 对比学习适配器
        self.null_contrast_proj = nn.Linear(self.d_model, self.d_model)
        # 默认text参数
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, d_model))
        # 对比学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0/contrastive_temp))
        self.motion_proj = nn.ModuleDict(
            {name: nn.Linear(self.d_model, parts_output_dim[name]) for name in self.original_vqvae.parts_name}
        )
        self.contrastive_loss = ContrastiveLossWithSTS(temperature=contrastive_temp)

    def forward(self, motion, text=None):
        if text is None:
            # 生成默认文本特征
            B = motion[0].shape[0]
            text_feature = self.null_contrast_proj(
                self.null_text_embed.expand(B, -1, -1)
            )
        else:
            text_feature = self.text_proj(text.pooler_output)
        # 特征增强
        enhanced_feat, fused_feat = self.cmt(motion)  # [B, T, part, d_model]
        # 对比学习损失
        motion_feature_global = self.motion_text_proj(fused_feat.mean(dim=1))  # [B, d_model]
        text_feature_global = text_feature  # [B, d_model]
        contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_global, text.pooler_output)
        # 原始编码流程
        # with torch.no_grad():  # 冻结原始编码器
        x_out_list = []
        loss_list = []
        perplexity_list = []
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            quantizer = getattr(self.original_vqvae, f'quantizer_{name}')
            decoder = getattr(self.original_vqvae, f'dec_{name}')
            x = motion[idx]
            x_in = self.original_vqvae.preprocess(x)
            x_encoder = encoder(x_in)
            x_encoder = x_encoder + self.original_vqvae.postprocess(self.motion_proj[name](enhanced_feat[:,:,idx,:]))
            x_quantized, code_idx, loss, perplexity = quantizer(x_encoder, sample_codebook_temp=0.5)
            x_decoder = decoder(x_quantized)
            x_out_list.append(self.original_vqvae.postprocess(x_decoder))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list, contrastive_loss
        
class EnhancedVQVAEv2(nn.Module):
    def __init__(self, args,
                 original_vqvae: VQVAE_ori,
                 text_encoder,
                 parts_output_dim,
                 d_model=256,
                 contrastive_temp=0.07):
        super().__init__()
        # 原始组件
        self.original_vqvae = original_vqvae
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusionV2(args, d_model=self.d_model, part_dims=part_dims)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)
        # 对比学习适配器
        self.null_contrast_proj = nn.Linear(self.d_model, self.d_model)
        # 默认text参数
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, d_model))
        # 对比学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0/contrastive_temp))
        self.motion_proj = nn.ModuleDict(
            {name: nn.Linear(self.d_model, parts_output_dim[name]) for name in self.original_vqvae.parts_name}
        )

        self.contrastive_loss = ContrastiveLossWithSTS(temperature=contrastive_temp)
        # 定义新的decoder, 输入shape为 parts_output_dim[name] + d_model
        for idx, name in enumerate(self.original_vqvae.parts_name):
            decoder = Decoder(original_vqvae.parts_input_dim[name], parts_output_dim[name] + d_model, down_t=original_vqvae.down_t, stride_t=original_vqvae.stride_t,  depth=original_vqvae.depth, dilation_growth_rate=original_vqvae.dilation_growth_rate, activation=original_vqvae.activation, norm=original_vqvae.norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        if text is None:
            # 生成默认文本特征
            B = motion[0].shape[0]
            text_feature = self.null_contrast_proj(
                self.null_text_embed.expand(B, -1, -1)
            )
        else:
            text_feature = self.text_proj(text.pooler_output)
        # 特征增强
        fused_feat, loss_cmt, perplexity_cmt = self.cmt(motion)  # [B, T, part, d_model]
        # 对比学习损失
        motion_feature_global = self.motion_text_proj(rearrange(fused_feat, 'b t p d -> b (p t) d', p=6).mean(dim=1))  # [B, d_model]
        text_feature_global = text_feature  # [B, d_model]
        contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_global, text.pooler_output)
        # 原始编码流程
        x_out_list = []
        loss_list = [loss_cmt]
        perplexity_list = [perplexity_cmt]
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            quantizer = getattr(self.original_vqvae, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x = motion[idx]
            x_in = self.original_vqvae.preprocess(x)
            x_encoder = encoder(x_in)
            x_encoder = x_encoder + self.original_vqvae.postprocess(self.motion_proj[name](fused_feat[:,:,idx,:]))
            x_quantized, loss, perplexity = quantizer(x_encoder)
            x_decoder = decoder(torch.cat((x_quantized, self.original_vqvae.postprocess(fused_feat[:,:,idx,:])),dim=1))
            x_out_list.append(self.original_vqvae.postprocess(x_decoder))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list, contrastive_loss        

class EnhancedVQVAEv3(nn.Module):
    def __init__(self, args,
                 original_vqvae: VQVAE_bodypart,
                 text_encoder,
                 parts_output_dim,
                 d_model=256,
                 contrastive_temp=0.07):
        super().__init__()
        # 原始组件
        self.original_vqvae = original_vqvae
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusionV2(args, d_model=self.d_model, part_dims=part_dims)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)
        # 对比学习适配器
        self.null_contrast_proj = nn.Linear(self.d_model, self.d_model)
        # 默认text参数
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, d_model))
        # 对比学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0/contrastive_temp))
        self.motion_proj = nn.ModuleDict(
            {name: nn.Linear(self.d_model, parts_output_dim[name]) for name in self.original_vqvae.parts_name}
        )
        self.encoder_proj = nn.ModuleDict(
            {name: nn.Conv1d(in_channels=parts_output_dim[name]*2, out_channels=parts_output_dim[name], kernel_size=1) for name in self.original_vqvae.parts_name}
        )
        self.contrastive_loss = ContrastiveLossWithSTS(temperature=contrastive_temp)
        # 定义新的decoder, 输入shape为 parts_output_dim[name] + d_model
        for idx, name in enumerate(self.original_vqvae.parts_name):
            decoder = Decoder(original_vqvae.parts_input_dim[name], parts_output_dim[name] + d_model, down_t=original_vqvae.down_t, stride_t=original_vqvae.stride_t, width=original_vqvae.parts_hidden_dim[name], depth=original_vqvae.depth, dilation_growth_rate=original_vqvae.dilation_growth_rate, activation=original_vqvae.activation, norm=original_vqvae.norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        if text is None:
            # 生成默认文本特征
            B = motion[0].shape[0]
            text_feature = self.null_contrast_proj(
                self.null_text_embed.expand(B, -1, -1)
            )
        else:
            text_feature = self.text_proj(text.pooler_output)
        # 特征增强
        fused_feat, loss_cmt, perplexity_cmt = self.cmt(motion)  # [B, T, part, d_model]
        # 对比学习损失
        motion_feature_global = self.motion_text_proj(rearrange(fused_feat, 'b t p d -> b (p t) d', p=6).mean(dim=1))  # [B, d_model]
        text_feature_global = text_feature  # [B, d_model]
        contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_global, text.pooler_output)
        # 原始编码流程
        x_out_list = []
        loss_list = [loss_cmt]
        perplexity_list = [perplexity_cmt]
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            quantizer = getattr(self.original_vqvae, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x = motion[idx]
            x_in = self.original_vqvae.preprocess(x)
            x_encoder = encoder(x_in)
            x_encoder = self.encoder_proj[name](torch.cat((x_encoder, self.original_vqvae.postprocess(self.motion_proj[name](fused_feat[:,:,idx,:]))),dim = 1))
            x_quantized, code_index, loss, perplexity = quantizer(x_encoder, sample_codebook_temp=0.5)
            x_decoder = decoder(torch.cat((x_quantized, self.original_vqvae.postprocess(fused_feat[:,:,idx,:])),dim=1))
            x_out_list.append(self.original_vqvae.postprocess(x_decoder))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list, contrastive_loss        

class EnhancedVQVAEv4(nn.Module):
    def __init__(self, args,
                 original_vqvae: VQVAE_bodypart,
                 parts_output_dim,
                 d_model=256,
                 contrastive_temp=0.07):
        super().__init__()
        # 原始组件
        self.original_vqvae = original_vqvae
        self.args = args
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusionV4(d_model=self.d_model, part_dims=part_dims)
        self.motion_proj = nn.ModuleDict(
            {name: nn.Linear(self.d_model, parts_output_dim[name]) for name in self.original_vqvae.parts_name}
        )
        self.encoder_proj = nn.ModuleDict(
            {name: nn.Conv1d(in_channels=parts_output_dim[name]*2, out_channels=parts_output_dim[name], kernel_size=1) for name in self.original_vqvae.parts_name}
        )
        self.contrastive_loss = ContrastiveLossWithSTS(temperature=contrastive_temp)
        # 定义新的decoder, 输入shape为 parts_output_dim[name] + d_model
        for idx, name in enumerate(self.original_vqvae.parts_name):
            if args.decoder_vision == 1:
                decoder = Decoder(original_vqvae.parts_input_dim[name], parts_output_dim[name] + d_model, down_t=original_vqvae.down_t, stride_t=original_vqvae.stride_t, width=original_vqvae.parts_hidden_dim[name], depth=original_vqvae.depth, dilation_growth_rate=original_vqvae.dilation_growth_rate, activation=original_vqvae.activation, norm=original_vqvae.norm)
            elif args.decoder_vision == 2:
                decoder = EnhancedDecoder(d_model, original_vqvae.parts_input_dim[name], parts_output_dim[name], down_t=original_vqvae.down_t, stride_t=original_vqvae.stride_t, width=original_vqvae.parts_hidden_dim[name], depth=original_vqvae.depth, dilation_growth_rate=original_vqvae.dilation_growth_rate, activation=original_vqvae.activation, norm=original_vqvae.norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        # 特征增强
        fused_feat = self.cmt(motion)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            quantizer = getattr(self.original_vqvae, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x = motion[idx]
            x_in = self.original_vqvae.preprocess(x)
            x_encoder = encoder(x_in)
            x_encoder = self.encoder_proj[name](torch.cat((x_encoder, self.original_vqvae.postprocess(self.motion_proj[name](fused_feat))),dim = 1))
            if self.args.num_quantizers == 1:
                x_quantized, loss, perplexity = quantizer(x_encoder)
            else:
                x_quantized, code_index, loss, perplexity = quantizer(x_encoder, sample_codebook_temp=0.5)
            if self.args.decoder_vision == 1:
                x_decoder = decoder(torch.cat((x_quantized, self.original_vqvae.postprocess(fused_feat)),dim=1))
            elif self.args.decoder_vision == 2:
                x_decoder = decoder(x_quantized, self.original_vqvae.postprocess(fused_feat))
            x_out_list.append(self.original_vqvae.postprocess(x_decoder))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list  

class EnhancedVQVAEv5(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        # 原始组件
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.part_dims = part_dims
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusionV5(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            if args.dataname == 't2m':
                self.parts_input_dim = {
                'Root': 7,
                'R_Leg': 50,
                'L_Leg': 50,
                'Backbone': 60,
                'R_Arm': 60,
                'L_Arm': 60,
                }
            elif args.dataname == 'kit':
                self.parts_input_dim = {
                'Root': 7,
                'R_Leg': 62,
                'L_Leg': 62,
                'Backbone': 48,
                'R_Arm': 48,
                'L_Arm': 48,
                }
            decoder = Decoder(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm, with_attn=args.with_attn)
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'dec_{name}', decoder)
            setattr(self, f'quantizer_{name}', quantizer)

    def forward(self, motion, text=None):
        # 特征增强
        global_feature, fused_feat = self.cmt(motion)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            if torch.any(torch.isnan(x_quantized)):
                print('decoder output has nan')
            x_decoder = decoder(x_quantized)
            if torch.any(torch.isnan(x_decoder)):
                print('decoder output has nan')
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list 

class EnhancedVQVAEv6(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV6(d_model=d_model, part_dims=self.part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        return super().forward(motion, text)
    
    def encode(self, motion, motion_mask=None):
        fused_feat = self.cmt(motion, motion_mask)
        if isinstance(fused_feat, tuple):
            fused_feat = torch.cat([fused_feat[0].unsqueeze(0), fused_feat[1]],dim=0)
        code_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            x_encoder = fused_feat[idx, ...]
            code_idx = quantizer.quantize(x_encoder)
            # code_idx[~motion_mask.bool()] = quantizer.nb_code
            if motion_mask is not None:
                if len(motion_mask.shape) == 2:
                    motion_mask = motion_mask[0]
                bool_mask = motion_mask.bool()
                motion_length = bool_mask.sum()
                if motion_length.item() < code_idx.shape[1]:
                    # padding
                    code_idx[:, ~bool_mask] = quantizer.nb_code + 1
                    # end token
                    code_idx[:, motion_length.item()] = quantizer.nb_code
            code_list.append(code_idx)
        return code_list

# 没有时间降采样的V5版本
class EnhancedVQVAEv7(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        # 新增模块
        self.cmt = EnhancedPartFusionV7(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

# 采用Gesture Transformer的版本
class EnhancedVQVAEv8(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        # 新增模块
        self.cmt = EnhancedPartFusionV8(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

# 采用Gesture Transformer的版本，完全版本
class EnhancedVQVAEv9(EnhancedVQVAEv8):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        # 新增模块
        self.cmt = EnhancedPartFusionV9(causal=args.causal ,d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv10(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV10(d_model=args.d_model, part_dims=self.part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv11(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        # 原始组件
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.part_dims = part_dims
        self.d_model = d_model
        # 新增模块
        # self.cmt = EnhancedPartFusionV5(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            if args.dataname == 't2m':
                self.parts_input_dim = {
                'Root': 7,
                'R_Leg': 50,
                'L_Leg': 50,
                'Backbone': 60,
                'R_Arm': 60,
                'L_Arm': 60,
                }
            elif args.dataname == 'kit':
                self.parts_input_dim = {
                'Root': 7,
                'R_Leg': 62,
                'L_Leg': 62,
                'Backbone': 48,
                'R_Arm': 48,
                'L_Arm': 48,
                }
            # encoder降采样为0
            encoder = Encoder(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=1, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm, with_attn=args.with_attn)
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'dec_{name}', decoder)
            setattr(self, f'quantizer_{name}', quantizer)
            setattr(self, f'enc_{name}', encoder)

    def forward(self, motion, text=None):
        # 特征增强
        # global_feature, fused_feat = self.cmt(motion)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        for idx, name in enumerate(self.parts_name):
            encoder = getattr(self, f'enc_{name}')
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            # x_encoder = fused_feat[idx, ...]
            x_encoder = encoder(motion[idx].float().permute(0,2,1))
            x_quantized, loss, perplexity = quantizer(x_encoder)
            x_decoder = decoder(x_quantized)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list

    def encode(self, motion, motion_mask=None):
        # fused_feat = self.cmt(motion, motion_mask)
        code_list = []
        for idx, name in enumerate(self.parts_name):
            encoder = getattr(self, f'enc_{name}')
            quantizer = getattr(self, f'quantizer_{name}')
            x_encoder = encoder(motion[idx].float().permute(0,2,1))
            code_idx = quantizer.quantize(x_encoder.permute(0,2,1))
            # code_idx[~motion_mask.bool()] = quantizer.nb_code
            if motion_mask is not None:
                if len(motion_mask.shape) == 2:
                    motion_mask = motion_mask[0]
                bool_mask = motion_mask.bool()
                motion_length = bool_mask.sum()
                
                # end token
                if motion_length.item() < code_idx.shape[1]:
                    # padding
                    code_idx[:, ~bool_mask] = quantizer.nb_code + 1
                    code_idx[:, motion_length.item()] = quantizer.nb_code
                    
            code_list.append(code_idx)
        return code_list
        

class EnhancedVQVAEv12(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV12(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv13(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV13(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers, position=args.position, causal=args.causal)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv14(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV14(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers, position=args.position, causal=args.causal)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv15(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV15(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers, position=args.position, causal=args.causal)

    def forward(self, motion, text=None):
        return super().forward(motion, text)
    
class EnhancedVQVAEv16(EnhancedVQVAEv5):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args, d_model=d_model)
        self.cmt = EnhancedPartFusionV16(d_model=self.d_model, part_dims=self.part_dims, num_layers=args.num_layers, position=args.position, causal=args.causal)

    def forward(self, motion, text=None):
        return super().forward(motion, text)

class EnhancedVQVAEv17(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        # 原始组件
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.part_dims = part_dims
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusionV17(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        self.sem_quantizer = QuantizeEMAReset(args.vqvae_sem_nb, d_model, args)
        # 对比学习对齐
        self.text_proj = nn.Linear(args.text_dim, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)
        self.post_quant_conv = nn.Conv1d(in_channels=d_model * 2, out_channels=d_model, kernel_size=1)
        if args.dataname == 't2m':
                self.parts_input_dim = {
                'Root': 7,
                'R_Leg': 50,
                'L_Leg': 50,
                'Backbone': 60,
                'R_Arm': 60,
                'L_Arm': 60,
                }
        elif args.dataname == 'kit':
            self.parts_input_dim = {
            'Root': 7,
            'R_Leg': 62,
            'L_Leg': 62,
            'Backbone': 48,
            'R_Arm': 48,
            'L_Arm': 48,
            }
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        for idx, name in enumerate(self.parts_name):
            decoder = Decoder_wo_upsample(self.parts_input_dim[name], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vqdec_norm, with_attn=args.with_attn)
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'dec_{name}', decoder)
            setattr(self, f'quantizer_{name}', quantizer)

    def forward(self, motion, text=None):
        # 特征增强
        global_feature, fused_feat = self.cmt(motion)  # [B, seq, d_model]
        global_quantized, global_loss, global_perplexity = self.sem_quantizer(rearrange(global_feature, 'b t d -> b d t'))
        # global_quantized = rearrange(global_quantized, 'b d t -> b t d')
        if text is not None:
            text_feature, text_id = text
            text_feature = text_feature.to(motion[0].device).float()
            text_feature = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_quantized.mean(dim=2))  # [B, d_model]
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion[0].device)
        
        # 原始编码流程
        x_out_list = []
        loss_list = [global_loss]
        perplexity_list = [global_perplexity]
        disentangle_loss = []
        # loss_extend = {}
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_decoder_fused = self.post_quant_conv(torch.cat([x_quantized, global_quantized], dim=1))
            x_decoder = decoder(x_decoder_fused)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            disentangle_loss.append(self.contrastive_loss.compute_disentangle_loss(x_quantized, global_quantized))
        
        return x_out_list, loss_list, perplexity_list, [contrastive_loss, disentangle_loss]

class EnhancedVQVAEv18(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        self.cmt = EnhancedPartFusionV6(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        self.decoder = PureMotionDecoder(d_model=self.d_model, output_dim=part_dims, num_layers=args.numdec_layers, with_attn=args.with_attn, with_global=args.with_global)
        self.sem_quantizer = QuantizeEMAReset(args.vqvae_sem_nb, d_model, args)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)

    def forward(self, motion, text=None):
        # 特征增强
        global_feature, fused_feat = self.cmt(motion)  # [B, seq, d_model]
        global_feature = rearrange(global_feature, 'b t d -> b d t')
        global_quantized, global_loss, global_perplexity = self.sem_quantizer(global_feature)
        # 原始编码流程
        x_out_list = []
        loss_list = [global_loss]
        perplexity_list = [global_perplexity]
        x_quantized_list = [global_quantized]
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            # decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized)
            # x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        x_out_list = self.decoder(x_quantized_list)
            # x_decoder = decoder(x_quantized)
            
        
        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0).to(motion[0].device)

class EnhancedVQVAEv19(EnhancedVQVAEv18):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args=args, d_model=d_model)
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.text_proj = nn.Linear(args.text_dim, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)
        self.post_quant_conv = nn.Conv1d(in_channels=d_model * 2, out_channels=d_model, kernel_size=1)

    def forward(self, motion, text=None):
        # 特征增强
        global_feature, fused_feat = self.cmt(motion)  # [B, seq, d_model]
        global_feature = rearrange(global_feature, 'b t d -> b d t')
        global_quantized, global_loss, global_perplexity = self.sem_quantizer(global_feature)
        if text is not None:
            text_feature, text_id = text
            text_feature = text_feature.to(motion[0].device).float()
            text_feature = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_quantized.mean(dim=2))  # [B, d_model]
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion[0].device)
        # 原始编码流程
        x_out_list = []
        loss_list = [global_loss]
        perplexity_list = [global_perplexity]
        x_quantized_list = [global_quantized]
        disentangle_loss = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            # decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized)
            disentangle_loss.append(self.contrastive_loss.compute_disentangle_loss(x_quantized, global_quantized))
            # x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        x_out_list = self.decoder(x_quantized_list)
            # x_decoder = decoder(x_quantized)
            
        
        return x_out_list, loss_list, perplexity_list, [contrastive_loss, disentangle_loss]

class EnhancedVQVAEv20(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        self.cmt = EnhancedPartFusionV20(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        if args.lgvq==1:
            self.lgvq = LGVQ(args, d_model=d_model, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)
            decoder = Decoder_wo_upsample(part_dims[idx], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        # 特征增强
        fused_feat = self.cmt(motion)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        x_quantized_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            x_decoder = decoder(x_quantized)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
        if self.args.lgvq>=1:
            _, contrastive_loss = self.lgvq(x_quantized_list, text)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion[0].device)
        return x_out_list, loss_list, perplexity_list, contrastive_loss

class EnhancedVQVAEv21(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        self.cmt = EnhancedPartFusionV20(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        if args.lgvq==1:
            self.lgvq = LGVQv2(args, d_model=d_model, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)
            decoder = Decoder_wo_upsample(part_dims[idx], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        if self.args.lgvq>=1 and text is not None and len(text) == 4:
            text_feature, text_id, text_mask, motion_mask = text
        else:
            motion_mask, text_mask = None, None
        # 特征增强
        fused_feat = self.cmt(motion, motion_mask)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        x_quantized_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            x_decoder = decoder(x_quantized)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
        if self.args.lgvq>=1 and len(text) == 4:
            # text_feature, text_id, text_mask, motion_mask = text
            _, loss = self.lgvq(x_quantized_list, [text_feature, text_id], text_mask, motion_mask)
        else:
            loss = torch.tensor(0.0).to(motion[0].device)
        return x_out_list, loss_list, perplexity_list, loss
    
    def encode(self, motion, motion_mask=None):
        fused_feat = self.cmt(motion, motion_mask)
        code_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            x_encoder = fused_feat[idx, ...]
            code_idx = quantizer.quantize(x_encoder)
            # code_idx[~motion_mask.bool()] = quantizer.nb_code
            if motion_mask is not None:
                if len(motion_mask.shape) == 2:
                    motion_mask = motion_mask[0]
                bool_mask = motion_mask.bool()
                motion_length = bool_mask.sum()
                if motion_length.item() < code_idx.shape[1]:
                    # padding
                    code_idx[:, ~bool_mask] = quantizer.nb_code + 1
                    # end token
                    code_idx[:, motion_length.item()] = quantizer.nb_code
            code_list.append(code_idx)
        return code_list

class EnhancedVQVAEv22(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        self.cmt = EnhancedPartFusionV6(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        self.dual = Dualsem_encoder(self.args, num_layers=self.args.num_layers, d_model=self.args.d_model)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)
            decoder = Decoder_wo_upsample(part_dims[idx], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        if text is not None and len(text) == 4:
            text_feature, text_id, text_mask, motion_mask = text
        else:
            text_feature, text_id, motion_mask, text_mask = None, None, None, None
        # 特征增强
        fused_feat = self.cmt(motion, motion_mask)  # [B, seq, d_model]
        if isinstance(fused_feat, tuple):
            fused_feat = torch.cat([fused_feat[0].unsqueeze(0), fused_feat[1]],dim=0)
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        x_quantized_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            x_decoder = decoder(x_quantized)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
        if len(text) == 4:
            _, loss = self.dual(fused_feat, [text_feature, text_id], text_mask, motion_mask)
        else:
            loss = torch.tensor(0.0).to(motion[0].device)
        return x_out_list, loss_list, perplexity_list, loss


class EnhancedVQVAEv23(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        self.cmt = EnhancedPartFusionV5p(d_model=self.d_model, part_dims=part_dims, num_layers=args.num_layers)
        if args.lgvq==1:
            self.lgvq = LGVQ(args, d_model=d_model, num_layers=args.num_layers)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)
            decoder = Decoder_wo_upsample(part_dims[idx], d_model, down_t=args.down_t, stride_t=args.stride_t, width=args.vqvae_arch_cfg['parts_hidden_dim'][name], depth=args.depth, dilation_growth_rate=args.dilation_growth_rate, activation=args.vq_act, norm=args.vq_norm)
            setattr(self, f'dec_{name}', decoder)

    def forward(self, motion, text=None):
        # 特征增强
        fused_feat = self.cmt(motion)  # [B, seq, d_model]
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        x_quantized_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            x_decoder = decoder(x_quantized)
            x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
        if self.args.lgvq>=1:
            _, contrastive_loss = self.lgvq(x_quantized_list, text)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion[0].device)
        return x_out_list, loss_list, perplexity_list, contrastive_loss

class EnhancedVQVAEv24(EnhancedVQVAEv21):
    def __init__(self, args,
                 d_model=256,
                 ):
        super().__init__(args=args, d_model=d_model)
        if args.lgvq==1:
            self.lgvq = LGVQv3(args, d_model=d_model, num_layers=args.num_layers)
        elif args.lgvq==4:
            self.lgvq = LGVQv4(args, d_model=d_model, num_layers=args.num_layers)
        elif args.lgvq==5:
            self.lgvq = LGVQv5(args, d_model=d_model, num_layers=args.lglayers)
    
    def forward(self, motion, text=None):
        return super().forward(motion, text)
    
    def encode(self, motion, motion_mask=None):
        return super().encode(motion, motion_mask)
    
    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        fused_feat = self.cmt(motion, motion_mask)
        # 原始编码流程
        x_out_list = []
        loss_list = []
        perplexity_list = []
        x_quantized_list = []
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            # decoder = getattr(self, f'dec_{name}')
            x_encoder = fused_feat[idx, ...]
            x_quantized, loss, perplexity = quantizer(rearrange(x_encoder, 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            # x_decoder = decoder(x_quantized)
            # x_out_list.append(rearrange(x_decoder, 'b d t -> b t d'))
            # text_feature, text_id, text_mask, motion_mask = text
        result = self.lgvq.text_motion_topk(x_quantized_list, text, motion_mask, topk, text_mask)
        return result

class HumanVQVAETransformer(nn.Module):
    def __init__(self,
                 args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.parts_output_dim = parts_output_dim
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_bodypart(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, down_t, stride_t, depth, dilation_growth_rate, activation=activation, norm=norm)
        # for param in self.vqvae.parameters():
        #     param.requires_grad = False
        self.enhancedvqvae = EnhancedVQVAE(args, self.vqvae, self.text_encoder, parts_output_dim=parts_output_dim)

    def encode(self, x, motion_mask = None):
        quants = self.vqvae.encode(x, motion_mask)
        return quants

    def forward(self, x, caption = None):
        if caption is not None:
            caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
            caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list, contrastive_loss = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, contrastive_loss

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out

    def forward_decoder_batch(self, x):
        x_out = self.vqvae.forward_decoder_batch(x)
        return x_out
        
    def load_checkpoint(self, checkpoint):
        param = torch.load(checkpoint)
        self.load_state_dict(param['net'], strict=False)
        print("load checkpoint from: ", checkpoint)

class HumanVQVAETransformerV2(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.vqvae = VQVAE_ori(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv2(args, self.vqvae, self.text_encoder, parts_output_dim=self.parts_output_dim)

    def forward(self, x, caption = None):
        if caption is not None:
            caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
            caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list, contrastive_loss = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, contrastive_loss
    
    def load_vqvae(self, checkpoint):
        param = torch.load(checkpoint)
        self.load_state_dict(param['net'], strict=False)
        print("load checkpoint from: ", checkpoint)
        
    def load_without_vqvae(self, checkpoint):
        param = torch.load(checkpoint)
        new_state_dict = {}
        for key, value in param['net'].items():
            if 'vqvae' in key and 'enhancedvqvae' not in key:
                # self.cmt.load_state_dict(value)
                continue
            else:
                new_state_dict[key] = value
        self.load_state_dict(new_state_dict, strict=False)
        print("load checkpoint from: ", checkpoint)

class HumanVQVAETransformerV3(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # self.vqvae = VQVAE_bodypart(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv3(args, self.vqvae, self.text_encoder, parts_output_dim=self.parts_output_dim)

    def forward(self, x, caption = None):
        if caption is not None:
            caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
            caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list, contrastive_loss = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, contrastive_loss

    def load_without_vqvae(self, checkpoint):
        param = torch.load(checkpoint)
        new_state_dict = {}
        for key, value in param['net'].items():
            if 'vqvae' in key and 'enhancedvqvae' not in key:
                # self.cmt.load_state_dict(value)
                continue
            elif 'enhancedvqvae.dec_' in key or 'enhancedvqvae.original_vqvae' in key:
                continue
            else:
                new_state_dict[key] = value
        self.load_state_dict(new_state_dict, strict=False)
        print("load checkpoint from: ", checkpoint)

class HumanVQVAETransformerV4(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        if args.num_quantizers == 1:
            self.vqvae = VQVAE_ori(args, **kwargs)
            # part_dims=[7,50,50,60,60,60]
        # self.vqvae = VQVAE_bodypart(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv4(args, self.vqvae, parts_output_dim=self.parts_output_dim)
        del self.tokenizer, self.text_encoder

    def forward(self, x, caption = None):
        # if caption is not None:
        #     caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
        #     caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)
    
    def load_without_vqvae(self, checkpoint):
        param = torch.load(checkpoint)
        new_state_dict = {}
        for key, value in param['net'].items():
            if 'vqvae' in key and 'enhancedvqvae' not in key:
                # self.cmt.load_state_dict(value)
                continue
            elif 'enhancedvqvae.dec_' in key or 'enhancedvqvae.original_vqvae' in key:
                continue
            else:
                new_state_dict[key] = value
        self.load_state_dict(new_state_dict, strict=False)
        print("load checkpoint from: ", checkpoint)

# 完全使用transformer的encoder
class HumanVQVAETransformerV5(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv5(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        # if caption is not None:
        #     caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
        #     caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)
    
    def load_without_vqvae(self, checkpoint):
        param = torch.load(checkpoint)
        new_state_dict = {}
        for key, value in param['net'].items():
            if 'vqvae' in key and 'enhancedvqvae' not in key:
                # self.cmt.load_state_dict(value)
                continue
            elif 'enhancedvqvae.dec_' in key or 'enhancedvqvae.original_vqvae' in key:
                continue
            else:
                new_state_dict[key] = value
        self.load_state_dict(new_state_dict, strict=False)
        print("load checkpoint from: ", checkpoint)

# 完全使用transformer的encoder
class HumanVQVAETransformerV6(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv6(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)
    
    def encode(self, x, motion_mask=None):
        return self.enhancedvqvae.encode(x, motion_mask)

# 没有降采样的V5版本
class HumanVQVAETransformerV7(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv7(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# Gesture Transformer版本
class HumanVQVAETransformerV8(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv8(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# Gesture Transformer完全版本
class HumanVQVAETransformerV9(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv9(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)
    

# Gesture Transformer完全版本
class HumanVQVAETransformerV10(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv10(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

class HumanVQVAETransformerV11(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv11(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)
    
    def encode(self, x, motion_mask=None):
        return self.enhancedvqvae.encode(x, motion_mask)

# 完全使用transformer的encoder
class HumanVQVAETransformerV12(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv12(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# 只用时间交互的模型
class HumanVQVAETransformerV13(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv13(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# 只用时间交互的模型
class HumanVQVAETransformerV14(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv14(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# 只用时间交互的模型
class HumanVQVAETransformerV15(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv15(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

# 只用时间交互的模型
class HumanVQVAETransformerV16(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv16(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, torch.tensor(0.0)

class HumanVQVAETransformerV17(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv17(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV18(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv18(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV19(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv19(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV20(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv20(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV21(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv21(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

    def encode(self, motion, motion_mask=None):
        return self.enhancedvqvae.encode(motion, motion_mask)

class HumanVQVAETransformerV22(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv22(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV23(HumanVQVAETransformer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.enhancedvqvae = EnhancedVQVAEv23(args, args.d_model)
        del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

class HumanVQVAETransformerV24(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.enhancedvqvae = EnhancedVQVAEv24(args, args.d_model)
        # del self.tokenizer, self.text_encoder, self.vqvae

    def forward(self, x, caption = None):
        x_out_list, loss_list, perplexity_list, loss_extend = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list, loss_extend

    def encode(self, motion, motion_mask=None):
        return self.enhancedvqvae.encode(motion, motion_mask)
    
    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        return self.enhancedvqvae.text_motion_topk(motion, text, motion_mask, topk, text_mask)

def test_text():
    import clip
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_encoder_sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # text_features = model.encode_text(text)
    captions = [
    'A human walking backwards.',
    'A person is walking backwards.',
    'Someone walks in a circle counterclockwise',
    'A person walks a full counter-clockwise circle.',
    'A human performs a tight 90◦ curve to the right.',
    'A person walks a quarter circle clockwise with 4 steps.',
    'human goes backwards starting with left',
    'A person walks backwards.',
    'a person walks in a circle to the left side.',
    'trump',]
    clip_tokens = clip.tokenize(captions).to(device)
    text_features_clip = model.encode_text(clip_tokens)
    text_features_sbert = text_encoder_sbert.encode(captions, convert_to_tensor=True)
    
    text_embeds_sbert = F.normalize(text_features_sbert, p=2, dim=-1)
    sim_matrix_sbert = torch.mm(text_embeds_sbert, text_embeds_sbert.T)
    
    text_embeds_clip = F.normalize(text_features_clip, p=2, dim=-1)
    sim_matrix_clip = torch.mm(text_embeds_clip, text_embeds_clip.T)
    # 可视化相似度矩阵
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(sim_matrix_sbert.cpu().detach().numpy(), ax=axes[0], annot=True, fmt=".2f", cmap="viridis")
    axes[0].set_title("SBERT 相似度矩阵")
    sns.heatmap(sim_matrix_clip.cpu().detach().numpy(), ax=axes[1], annot=True, fmt=".2f", cmap="viridis")
    axes[1].set_title("CLIP 相似度矩阵")
    # plt.show()
    plt.savefig('text_sim.png')
    
    

class HumanVQVAEBodyPart(nn.Module):
    def __init__(self,
                 args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_bodypart(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, down_t, stride_t, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        quants = self.vqvae.encode(x)
        return quants

    def forward(self, x, caption = None):

        x_out_list, loss_list, perplexity_list = self.vqvae(x)

        return x_out_list, loss_list, perplexity_list

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out

    def forward_decoder_batch(self, x):
        x_out = self.vqvae.forward_decoder_batch(x)
        return x_out
        
    def load_checkpoint(self, checkpoint):
        param = torch.load(checkpoint)
        self.load_state_dict(param['vq_model'], strict=False)
        
def multi_part_similarity(motion, text):
    """
    motion: [batch, 6, seq_m, dim_m]  # 6个身体部位的动作特征
    text:   [batch, seq_t, dim_t]     # 文本特征
    
    返回: [batch] 相似度分数
    """
    # 维度统一方案（非学习方式）
    dim = min(motion.size(-1), text.size(-1))  # 取较小维度
    motion = motion[..., :dim]  # 截断动作维度
    text = text[..., :dim]       # 截断文本维度
    
    # 特征归一化
    motion_norm = F.normalize(motion, p=2, dim=-1)  # [B,6,Tm,d]
    text_norm = F.normalize(text, p=2, dim=-1)      # [B,Tt,d]
    
    # 多部位特征融合
    motion_pool = motion_norm.mean(dim=1)  # [B,Tm,d] 部位维度平均
    
    # 构造三维相似度矩阵
    sim_matrix = torch.einsum('btmd,btd->btm', 
                            motion_pool.unsqueeze(2),  # [B,Tm,1,d]
                            text_norm.unsqueeze(1))    # [B,1,Tt,d]
    
    # 双向最大对齐
    text_max = sim_matrix.max(dim=2)[0].mean(dim=1)  # 动作→文本 [B]
    motion_max = sim_matrix.max(dim=1)[0].mean(dim=1) # 文本→动作 [B]
    
    return (text_max + motion_max) / 2  # 双向平均        

def part_wise_similarity(motion, text):
    """
    motion: [batch, 6, seq_m, dim_m]  # 6个身体部位独立特征
    text:   [batch, seq_t, dim_t]     # 文本特征
    """
    # 维度统一（非学习方式）
    dim = min(motion.size(-1), text.size(-1))
    motion = motion[..., :dim]  # [B,6,Tm,d]
    text = text[..., :dim]       # [B,Tt,d]
    
    # 特征归一化
    motion = F.normalize(motion, p=2, dim=-1)  # 各部位独立归一化
    text = F.normalize(text, p=2, dim=-1)
    
    batch_size, num_parts = motion.shape[:2]
    sim_scores = torch.zeros(batch_size, num_parts, device=motion.device)
    
    # 对每个部位独立计算
    for part_idx in range(num_parts):
        # 当前部位特征 [B,Tm,d]
        part_motion = motion[:, part_idx]  
        
        # 三维相似度矩阵计算
        # [B,Tm,d] x [B,d,Tt] -> [B,Tm,Tt]
        sim_matrix = torch.einsum('bmd,bdt->bmt', 
                                part_motion, 
                                text.transpose(1,2))
        
        # 双向最大对齐
        text_max = sim_matrix.max(dim=2)[0].mean(dim=1)  # 部位→文本 [B]
        motion_max = sim_matrix.max(dim=1)[0].mean(dim=1) # 文本→部位 [B]
        
        # 记录当前部位相似度
        sim_scores[:, part_idx] = (text_max + motion_max) / 2
    
    # 合并各部位结果
    return sim_scores.mean(dim=1)  # [B] 各部位平均