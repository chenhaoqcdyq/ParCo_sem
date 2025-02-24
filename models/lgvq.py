import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import CLIPModel, CLIPTokenizer
from models.vqvae_bodypart import VQVAE_bodypart

class T2M_VQVAE_LG(VQVAE_bodypart):
    def __init__(self, args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None, 
                 text_dim=512,
                 num_heads=4,
                 num_layers=2,
                 clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, 
                         down_t, stride_t, depth, dilation_growth_rate, activation, norm)
        
        # Original VQVAE components
        # self.encoder = MotionEncoder(motion_dim, num_code)
        # self.decoder = MotionDecoder(motion_dim)
        # self.codebook = nn.Embedding(num_code, motion_dim)
        self.text_feat_dim = 512
        
        self.clip, self.tokenizer = self.load_clip(clip_model_name)

        self.vit = nn.ModuleDict(
            {name: MotionViT(self.parts_output_dim[name], num_layers, num_heads) for name in self.parts_output_dim}
        )
        self.cls_token = {
            name: nn.Parameter(torch.randn(1, 1, part_dim)) for name, part_dim in self.parts_output_dim.items()
        }
        # self.vit[name] = MotionViT(self.parts_output_dim[name], num_layers, num_heads)  # 1D ViT for motion codes
        # Masked text prediction
        # self.mask_decoder = CrossAttentionDecoder(motion_dim, text_dim)
        self.mask_decoder = nn.ModuleDict(
            {name: CrossAttentionDecoder(self.parts_output_dim[name], text_dim) for name in self.parts_output_dim}
        )
        self.relation_proj_text = nn.Sequential(
            nn.Linear(self.text_feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        self.relation_proj_motion = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.parts_output_dim[name], 256),
                nn.LayerNorm(256),
                nn.GELU()
            ) for name in self.parts_name}
        )
        self.gsa_text_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.text_feat_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, self.parts_output_dim[name]))  # 对齐主部位维度
            for name in self.parts_name
            }
        )
        
        # Loss weights
        self.alpha = 1.0  # GSA weight
        self.beta = 0.5   # MTP weight
        self.gamma = 0.2  # RAS weight

    def load_checkpoint(self, checkpoint):
        """加载预训练模型"""
        param = torch.load(checkpoint)
        new_param = {}
        for key, value in param['net'].items():
            if 'vqvae.' in key:
                new_key = key.replace('vqvae.', '')
                new_param[new_key] = value
        self.load_state_dict(new_param, strict=False)

    def load_clip(self, model_name):
        clip = CLIPModel.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # 冻结所有CLIP参数
        for param in clip.parameters():
            param.requires_grad = False
        return clip, tokenizer
    
    def _get_text_features(self, text):
        """获取文本特征"""
        with torch.no_grad():
            text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text_tokens = {k: v.to(self.clip.device) for k, v in text_tokens.items()}
            text_features = self.clip.get_text_features(**text_tokens)
        return text_tokens, text_features
    
    def forward(self, parts, text):
        """
        分身体前向传播
        :param parts: 身体部位特征列表 [Root, R_Leg, ...]
        :param text: 文本描述列表
        :return: 各模块输出
        """
        # 文本特征处理
        # text_tokens, text_feat = self._get_text_features(text)  # [B, text_dim]
        if isinstance(text, tuple):
            # text = text[0]
            text = [name for id, name in enumerate(text)]
        if isinstance(text, str):
            text = [text]
        if isinstance(text, list):
            text_tokens, text_feat = self._get_text_features(text)
        else:
            text_feat = text
        B = text_feat.size(0)
        
        # 存储各部位输出
        outputs = {
            'recon_parts': [],
            'vq_losses': [],
            'gsa_features': [],
            'mtp_logits': [],
            'relation_sims': [],
            'perplexity': []
        }
        
        # 遍历每个身体部位
        for i, name in enumerate(self.parts_name):
            # 原始VQVAE处理流程
            x_part = parts[i]
            x_in = self.preprocess(x_part)
            
            # 编码
            encoder = getattr(self, f'enc_{name}')
            z_e = encoder(x_in)  # [B, C, T]
            
            # 量化
            quantizer = getattr(self, f'quantizer_{name}')
            z_q, vq_loss, perplexity = quantizer(z_e)
            
            # 解码
            decoder = getattr(self, f'dec_{name}')
            recon_part = self.postprocess(decoder(z_q))
            
            outputs['recon_parts'].append(recon_part)
            outputs['vq_losses'].append(vq_loss)
            outputs['perplexity'].append(perplexity)
            # 语言引导模块 -------------------------------------------------
            # 添加CLS token并处理ViT
            cls_token = self.cls_token[name].expand(B, -1, -1).to(x_in.device)  # [B, 1, C]
            z_q_cls = torch.cat([cls_token, z_q.permute(0,2,1)], dim=1)  # [B, T+1, C]
            z_vt = self.vit[name](z_q_cls)  # [B, T+1, C]
            
            # 全局语义对齐特征
            e_CLS = z_vt[:, 0]  # [B, C]
            e_TEXT = self.gsa_text_proj[name](text_feat)
            outputs['gsa_features'].append((e_CLS, e_TEXT))
            
            # 掩码文本预测
            masked_feat = self._mask_text_feature(text_feat)
            pred_logits = self.mask_decoder[name](z_vt, masked_feat.unsqueeze(1))
            outputs['mtp_logits'].append((pred_logits.squeeze(1), text_feat))
            
            # 关系对齐
            word_emb = self.relation_proj_text(text_feat)  # [B, 256]
            code_emb = self.relation_proj_motion[name](z_vt[:, 1:].mean(dim=1))  # [B, 256]
            sim_matrix = F.cosine_similarity(word_emb.unsqueeze(1), code_emb.unsqueeze(0), dim=-1)
            outputs['relation_sims'].append(sim_matrix)
        
        return outputs

    def calculate_loss(self, outputs):
        """计算多任务损失"""
        # 基础VQ损失
        # vq_loss = torch.stack(outputs['vq_losses']).mean()
        # perplexity = torch.stack(outputs['perplexity']).mean()
        # 全局语义对齐损失 (对比学习)
        gsa_loss = 0
        for e_CLS, e_TEXT in outputs['gsa_features']:
            logits = torch.matmul(F.normalize(e_CLS, dim=-1), 
                                F.normalize(e_TEXT, dim=-1).t())  # [B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
            gsa_loss += F.cross_entropy(logits, labels)
        
        # 掩码文本预测损失 (特征回归)
        mtp_loss = 0
        for pred, target in outputs['mtp_logits']:
            mtp_loss += F.mse_loss(pred, target.detach())
        
        # 关系对齐损失
        ras_loss = 0
        for sim_matrix in outputs['relation_sims']:
            ras_matrix = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
            ras_loss += F.mse_loss(sim_matrix, ras_matrix)
        
        # 总损失
        total_loss = (self.alpha * gsa_loss +
                     self.beta * mtp_loss +
                     self.gamma * ras_loss)
        
        return {
            'total_loss': total_loss,
            # 'vq_loss': vq_loss,
            # 'perplexity': perplexity,
            'gsa_loss': gsa_loss,
            'mtp_loss': mtp_loss,
            'ras_loss': ras_loss
        }

    # # Helper functions
    # def quantize(self, z_e):
    #     # Vector quantization logic
    #     distances = torch.cdist(z_e, self.codebook.weight)
    #     indices = torch.argmin(distances, dim=-1)
    #     z_q = self.codebook(indices)
    #     return z_q, indices

    def _mask_text_feature(self, text_feat, mask_ratio=0.15):
        """生成掩码文本特征"""
        mask = torch.rand_like(text_feat) < mask_ratio
        return text_feat * ~mask  # 简单置零掩码

class MotionViT(nn.Module):
    # 1D Vision Transformer for processing motion codes
    def __init__(self, dim, num_layers, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads, dim*4)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CrossAttentionDecoder(nn.Module):
    def __init__(self, motion_dim, text_dim=512):  # CLIP默认文本维度
        super().__init__()
        
        # 维度对齐投影层
        self.motion_proj = nn.Linear(motion_dim, text_dim)
        
        # 跨注意力机制（使用文本维度）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, 4*text_dim),
            nn.GELU(),
            nn.Linear(4*text_dim, text_dim)
        )
        
        # 输出投影层（适配CLIP特征空间）
        self.proj = nn.Linear(text_dim, text_dim)  # 保持维度一致

    def forward(self, motion_feat, masked_text_feat):
        """
        motion_feat: [B, T+1, motion_dim] 运动特征（含CLS token）
        masked_text_feat: [B, S, text_dim] 被掩码的CLIP文本特征
        """
        # 运动特征维度投影
        projected_motion = self.motion_proj(motion_feat)  # [B, T+1, text_dim]
        
        # 跨注意力计算（Query: 文本， Key/Value: 运动）
        attn_out, _ = self.cross_attn(
            query=masked_text_feat,
            key=projected_motion,
            value=projected_motion
        )
        
        # 特征增强
        ffn_out = self.ffn(attn_out)
        
        # 最终投影（保持与CLIP特征相同维度）
        return self.proj(ffn_out)  # [B, S, text_dim]
    