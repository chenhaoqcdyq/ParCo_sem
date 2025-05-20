from einops import rearrange
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder
import torch.nn.functional as F
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.residual_vq import ResidualVQ
from transformers import CLIPTextModel, CLIPTokenizer  # 使用Hugging Face版本
import os
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

                if args.quantizer == "ema_reset":
                    quantizer = QuantizeEMAReset(nb_code, code_dim, args)
                elif args.quantizer == "orig":
                    quantizer = Quantizer(nb_code, code_dim, 1.0)
                elif args.quantizer == "ema":
                    quantizer = QuantizeEMA(nb_code, code_dim, args)
                elif args.quantizer == "reset":
                    quantizer = QuantizeReset(nb_code, code_dim, args)
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

                if args.quantizer == "ema_reset":
                    quantizer = QuantizeEMAReset(nb_code, code_dim, args)
                elif args.quantizer == "orig":
                    quantizer = Quantizer(nb_code, code_dim, 1.0)
                elif args.quantizer == "ema":
                    quantizer = QuantizeEMA(nb_code, code_dim, args)
                elif args.quantizer == "reset":
                    quantizer = QuantizeReset(nb_code, code_dim, args)
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


    def encode(self, parts):
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

            tokenized_parts.append(code_idx)

        return tokenized_parts


    def forward(self, parts):
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
            x_quantized, loss, perplexity = quantizer(x_encoder)

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

class EnhancedPartFusion(nn.Module):
    def __init__(self, args,
                 part_dims=[7,50,50,60,60,60],
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 use_zero_init=True):
        super().__init__()
        
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
        
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
            num_layers=num_layers
        )
        # 残差连接优化
        # self.res_conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.quantizer = QuantizeEMAReset(1024, d_model, args)
        # if use_zero_init:
        #     nn.init.zeros_(self.res_conv.weight)

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
        
        fused_feat = rearrange(spatial_feat, '(b t) p d -> (b t) d p', b=B, p=6)
        fused_feat, loss, perplexity = self.quantizer(fused_feat)
        fused_feat = rearrange(fused_feat, '(b t) d p -> b t p d', b=B, p=6)
        # 残差连接增强
        return fused_feat, loss, perplexity

class HierarchicalAttention(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        # 局部注意力（相邻关节）
        self.local_attn = nn.MultiheadAttention(d_model, nhead//2)
        # 全局注意力
        self.global_attn = nn.MultiheadAttention(d_model, nhead)
        
        # 基于运动链的注意力掩码
        self.register_buffer("mask", self.create_kinematic_mask())
    
    def create_kinematic_mask(self):
        """创建基于人体运动链的注意力掩码"""
        # 示例掩码，实际需根据具体骨骼结构定义
        attn_mask = [
            [1,1,1,1,1,1],  # Root与所有部位交互
            [1,1,0,1,0,0],  # R_Leg仅与Root、Backbone交互
            [1,0,1,1,0,0],  # L_Leg同理
            [1,1,1,1,1,1],  # Backbone全连接
            [1,0,0,1,1,0],  # R_Arm与Root、Backbone、自身交互
            [1,0,0,1,0,1]   # L_Arm同理
        ]
        return torch.Tensor(attn_mask)

    def forward(self, x):
        # 局部注意力（使用因果卷积获取邻域）
        local_feat, _ = self.local_attn(x, x, x)
        # 全局注意力（应用运动学约束）
        global_feat, _ = self.global_attn(x, x, x, attn_mask=self.mask)
        return local_feat + global_feat


class EnhancedVQVAE(nn.Module):
    def __init__(self, args,
                 original_vqvae: VQVAE_bodypart,
                #  text_encoder,
                 parts_output_dim,
                 d_model=256):
        super().__init__()
        # 原始组件
        self.original_vqvae = original_vqvae
        if args.dataname == 't2m':
            part_dims=[7,50,50,60,60,60]
        else:
            part_dims=[7,62,62,48,48,48]
        self.d_model = d_model
        # 新增模块
        self.cmt = EnhancedPartFusion(args, d_model=self.d_model, part_dims=part_dims)

        self.motion_proj = nn.ModuleDict(
            {name: nn.Linear(self.d_model, parts_output_dim[name]) for name in self.original_vqvae.parts_name}
        )
        
        # 冻结编码器
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            for param in encoder.parameters():
                param.requires_grad = False
            raw_dim = part_dims[idx]
            hidden_dim = original_vqvae.parts_hidden_dim[name]
            output_dim = parts_output_dim[name]
            decoder = Decoder(raw_dim, output_dim + d_model, original_vqvae.down_t, original_vqvae.stride_t, hidden_dim, original_vqvae.depth, original_vqvae.dilation_growth_rate, activation=original_vqvae.activation, norm=original_vqvae.norm)
            setattr(self, f'dec_{name}', decoder)
            conv1d = nn.Conv1d(in_channels=2*output_dim, out_channels=output_dim, kernel_size=1)
            setattr(self, f'enccov_{name}', conv1d)

    def forward(self, motion, text=None):
        # 特征增强
        fused_feat, loss_cmt, perplexity_cmt = self.cmt(motion)  # [B, T, part, d_model]
        # 原始编码流程
        # with torch.no_grad():  # 冻结原始编码器
        x_out_list = []
        loss_list = [loss_cmt]
        perplexity_list = [perplexity_cmt]
        for idx, name in enumerate(self.original_vqvae.parts_name):
            encoder = getattr(self.original_vqvae, f'enc_{name}')
            encoder_cov = getattr(self, f'enccov_{name}')
            quantizer = getattr(self.original_vqvae, f'quantizer_{name}')
            decoder = getattr(self, f'dec_{name}')
            x = motion[idx]
            x_in = self.original_vqvae.preprocess(x)
            x_encoder = encoder(x_in)
            x_encoder_cat = torch.cat((x_encoder , self.original_vqvae.postprocess(self.motion_proj[name](fused_feat[:,:,idx,:]))) , dim=1)
            x_encoder = encoder_cov(x_encoder_cat)
            x_quantized, loss, perplexity = quantizer(x_encoder)
            x_decoder_cat = torch.cat((x_quantized, self.original_vqvae.postprocess(fused_feat[:,:,idx,:])), dim=1)
            x_decoder = decoder(x_decoder_cat)
            x_out_list.append(self.original_vqvae.postprocess(x_decoder))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
        
        return x_out_list, loss_list, perplexity_list
        
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
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_bodypart(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, down_t, stride_t, depth, dilation_growth_rate, activation=activation, norm=norm)
        # for param in self.vqvae.parameters():
        #     param.requires_grad = False
        self.enhancedvqvae = EnhancedVQVAE(args, self.vqvae, parts_output_dim=parts_output_dim)

    def encode(self, x):
        quants = self.vqvae.encode(x)
        return quants

    def forward(self, x, caption = None):
        # if caption is not None:
        #     caption_tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(x[0].device)
        #     caption = self.text_encoder(**caption_tokens)
        x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)

        return x_out_list, loss_list, perplexity_list

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


class HumanTransformerV2(nn.Module):
    def __init__(self,
                 args,
                 global_code_dim=256,
                 detail_code_dim=128,
                 n_global_codes=512,
                 n_detail_codes=1024,
                 time_steps=196,
                 **kwargs):
        
        super().__init__()
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_bodypart(args, **kwargs)
        # ------------------ 编码器部分 ------------------
        # 全局语义编码器
        self.global_encoder = GlobalEncoder(self.nb_joints, global_code_dim)
        
        # 细节条件编码器
        # self.detail_encoder = DetailConditionalEncoder(global_code_dim, detail_code_dim)
        
        # ------------------ 量化层部分 ------------------
        # 全局语义码本
        self.global_codebook = QuantizeEMA(n_global_codes, global_code_dim)
        
        # ------------------ 解码器部分 ------------------
        self.decoder = HierarchicalDecoder(global_code_dim, detail_code_dim, self.nb_joints)
        
        # ------------------ 改进模块 ------------------
        self.temporal_align = TemporalAlign(global_code_dim, detail_code_dim)
        self.cross_attn = CrossPartAttention(global_code_dim)
        self.gate_mechanism = AdaptiveGating(global_code_dim, detail_code_dim)
        
        # ------------------ 配置参数 ------------------
        self.time_steps = time_steps
        # self.n_joints = n_joints
        # self.enhancedvqvae = EnhancedVQVAE(args, self.vqvae, parts_output_dim=parts_output_dim)

    def encode(self, x):
        quants = self.vqvae.encode(x)
        return quants

    def forward(self, x, caption = None):
        # x_out_list, loss_list, perplexity_list = self.enhancedvqvae(x, caption)
        B, T, D = x.shape  # [B, 196, 24]
        
        # ------------------ 全局语义编码 ------------------
        global_feat = self.global_encoder(x)  # [B, 49, 256]
        
        # 全局量化
        global_quant, global_loss, _ = self.global_codebook(global_feat)
        
        # ------------------ 细节编码 ------------------
        aligned_global = self.temporal_align(global_quant)  # [B, 196, 128]
        detail_feat = self.detail_encoder(x, aligned_global)  # [B, 196, 6, 128]
        
        # 分部位量化
        detail_quants = []
        detail_loss = 0
        for i in range(6):
            d_quant, d_loss, _ = self.detail_codebooks[i](detail_feat[:,:,:,i])
            detail_quants.append(d_quant)
            detail_loss += d_loss
        
        # ------------------ 特征融合 ------------------
        fused_feat = self.gate_mechanism(
            self.cross_attn(global_quant), 
            torch.stack(detail_quants, dim=2)
        )  # [B, 196, 6, 128]
        
        # ------------------ 运动重建 ------------------
        recon_motion = self.decoder(global_quant, fused_feat)  # [B, 196, 24]
        
        # ------------------ 损失计算 ------------------
        # recon_loss = F.mse_loss(x, recon_motion)
        # total_loss = recon_loss + 0.5*(global_loss + detail_loss/6)

        return recon_motion, loss_list, perplexity_list

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

class GlobalEncoder(nn.Module):
    """全局语义编码器"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            CausalConv1D(input_dim, 128, kernel_size=5),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.GroupNorm(8, 256)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv(x)        # [B, 256, 49]
        x = x.permute(0, 2, 1)  # [B, 49, 256]
        x = self.transformer(x)
        return self.fc(x)

# class DetailConditionalEncoder(nn.Module):
#     """细节条件编码器"""
#     def __init__(self, global_dim, output_dim):
#         super().__init__()
#         self.local_conv = nn.ModuleList([
#             nn.Sequential(
#                 CausalConv1D(4, 64, kernel_size=3),
#                 nn.GELU(),
#                 nn.Conv1d(64, output_dim, kernel_size=1)
#             ) for _ in range(6)  # 6个身体部位
#         ])
#         self.condition_proj = nn.Linear(global_dim, output_dim)
        
#     def forward(self, x, global_feat):
#         # x: [B, T, 24] -> [B, T, 6, 4]
#         x_parts = x.view(-1, self.time_steps, 6, 4).permute(0,2,3,1)
        
#         part_features = []
#         for i in range(6):
#             feat = self.local_conv[i](x_parts[:,i])  # [B, C, T]
#             part_features.append(feat.permute(0,2,1))
            
#         # 全局条件投影
#         cond_feat = self.condition_proj(global_feat)  # [B, T, C]
#         return torch.stack(part_features, dim=2) + cond_feat.unsqueeze(2)

class HierarchicalDecoder(nn.Module):
    """层次化解码器"""
    def __init__(self, global_dim, detail_dim, output_dim):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Conv1d(global_dim, 128, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.detail_convs = nn.ModuleList([
            nn.Conv1d(detail_dim, 128, kernel_size=3) 
            for _ in range(6)
        ])
        self.fusion_conv = nn.Conv1d(128 * 7, output_dim, kernel_size=1)
        
    def forward(self, global_codes, detail_codes):
        # global_codes: [B, T_g, C]
        # detail_codes: [B, T, 6, C]
        
        # 全局特征处理
        global_feat = self.global_proj(global_codes.permute(0,2,1))  # [B, C, T]
        
        # 细节特征处理
        detail_feats = []
        for i in range(6):
            feat = self.detail_convs[i](detail_codes[:,:,:,i].permute(0,2,1))
            detail_feats.append(feat)
            
        # 特征融合
        combined = torch.cat([global_feat] + detail_feats, dim=1)
        return self.fusion_conv(combined).permute(0,2,1)

# ------------------ 改进组件实现 ------------------

class CrossPartCoordinator(nn.Module):
    def __init__(self, part_dims, code_dim=256, n_heads=4):
        super().__init__()
        # 各部位特征投影
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, code_dim) 
            for dim in part_dims  # [7,50,50,60,60,60]
        ])
        
        # 运动学注意力机制
        self.attn = KinematicAttention(code_dim, n_heads)
        
        # 时间同步卷积
        self.temporal_conv = nn.Conv1d(
            in_channels=6*code_dim,
            out_channels=6*code_dim,
            kernel_size=3,
            padding=1,
            groups=6  # 分组卷积保持部位独立性
        )

    def forward(self, part_features):
        # 投影到统一空间
        projected = []
        for i, part in enumerate(['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):
            proj_feat = self.proj_layers[i](part_features[part]['z_e'])
            projected.append(proj_feat.unsqueeze(2))
            
        # 组合特征 [B, T, 6, D]
        combined = torch.cat(projected, dim=2)
        
        # 运动学注意力
        attn_out = self.attn(combined)
        
        # 时间维度特征增强
        temporal_feat = self.temporal_conv(
            attn_out.permute(0,3,1,2).flatten(1,2)
        ).view_as(attn_out)
        
        return attn_out + temporal_feat  # 残差连接

class FrameWiseGlobalEncoder(nn.Module):
    """帧级全局语义编码器"""
    def __init__(self, part_dims, code_dim):
        super().__init__()
        # 各部位特征投影
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, code_dim) for dim in part_dims
        ])
        
        # 跨部位注意力
        self.attn = nn.MultiheadAttention(code_dim, num_heads=8)
        
        # 时间因果卷积
        self.temporal_conv = CausalConv1D(code_dim, code_dim, kernel_size=5)
        
    def forward(self, x):
        """输入: [B, T, 6, part_dim]"""
        B, T = x.shape[:2]
        
        # 各部位特征投影
        projected = []
        for i in range(6):
            part_feat = x[:, :, i, :]
            proj_feat = self.proj_layers[i](part_feat)  # [B, T, C]
            projected.append(proj_feat)
            
        # 跨部位注意力
        combined = torch.stack(projected, dim=1)  # [B, 6, T, C]
        attn_out, _ = self.attn(
            combined.view(B*6, T, -1), 
            combined.view(B*6, T, -1), 
            combined.view(B*6, T, -1)
        )
        attn_out = attn_out.view(B, 6, T, -1).mean(1)  # [B, T, C]
        
        # 时间维度处理
        temporal_out = self.temporal_conv(attn_out.permute(0,2,1)).permute(0,2,1)
        return F.gelu(temporal_out + attn_out)  # 残差连接

class KinematicAttention(nn.Module):
    """基于人体运动学的注意力机制"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            kdim=dim,
            vdim=dim
        )
        # 运动链掩码矩阵
        self.register_buffer('mask', self._build_kinematic_mask())
        
    def _build_kinematic_mask(self):
        """定义允许交互的身体部位对"""
        # mask = torch.ones(6,6)
        # 禁止非对称部位直接交互
        mask = [
            [1,1,1,1,1,1],  # Root与所有部位交互
            [1,1,0,1,0,0],  # R_Leg仅与Root、Backbone交互
            [1,0,1,1,0,0],  # L_Leg同理
            [1,1,1,1,1,1],  # Backbone全连接
            [1,0,0,1,1,0],  # R_Arm与Root、Backbone、自身交互
            [1,0,0,1,0,1]   # L_Arm同理
        ]
        mask = torch.Tensor(mask)
        return mask.bool()
    
    def forward(self, x):
        # x: [B, T, 6, D]
        B, T, N, D = x.shape
        x = x.permute(1,0,2,3).flatten(0,1)  # [T*B, 6, D]
        
        # 注意力计算
        attn_out, _ = self.attention(
            query=x, key=x, value=x,
            key_padding_mask=~self.mask
        )
        
        return attn_out.view(T, B, N, D).permute(1,0,2,3)

# class CausalConv1D(nn.Conv1d):
#     """因果卷积实现"""
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         padding = (kernel_size - 1) * kwargs.get('dilation', 1)
#         super().__init__(in_channels, out_channels, kernel_size, 
#                         padding=padding, **kwargs)
        
#     def forward(self, x):
#         conv_out = super().forward(x)
#         return conv_out[:, :, :-self.padding[0]]

class CausalConv1D(nn.Conv1d):
    """因果卷积实现"""
    def __init__(self, in_channels, out_channels, kernel_size):
        padding = (kernel_size - 1) * 1
        super().__init__(in_channels, out_channels, kernel_size, 
                        padding=padding)
        
    def forward(self, x):
        x = super().forward(x)
        return x[:, :, :-self.padding[0]] if self.padding[0] !=0 else x

class TemporalAlign(nn.Module):
    """时间对齐模块"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='linear')
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        # x: [B, T_global, C]
        x_up = self.upsample(x.permute(0,2,1))  # [B, C, T_local]
        return self.conv(x_up).permute(0,2,1)

class CrossPartAttention(nn.Module):
    """跨部位注意力"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, T, 6, C]
        B, T, N, C = x.shape
        q = self.query(x).view(B*T, N, C)
        k = self.key(x).view(B*T, N, C)
        v = self.value(x).view(B*T, N, C)
        
        attn = F.softmax(torch.bmm(q, k.transpose(1,2))/torch.sqrt(C), dim=-1)
        return (torch.bmm(attn, v) + x).view(B, T, N, C)

class AdaptiveGating(nn.Module):
    """自适应门控机制"""
    def __init__(self, global_dim, detail_dim):
        super().__init__()
        self.gate_fc = nn.Linear(global_dim + detail_dim, detail_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, global_feat, detail_feat):
        # global_feat: [B, T, C]
        # detail_feat: [B, T, 6, C]
        gate_input = torch.cat([
            global_feat.unsqueeze(2).expand_as(detail_feat),
            detail_feat
        ], dim=-1)
        
        gate = self.sigmoid(self.gate_fc(gate_input))
        return gate * detail_feat