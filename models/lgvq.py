import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from transformers import CLIPModel, CLIPTokenizer
import clip
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
                 clip_model_name='ViT-B/32'):
        super().__init__(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, 
                         down_t, stride_t, depth, dilation_growth_rate, activation, norm)
        
        # Original VQVAE components
        self.text_feat_dim = 512
        
        self.clip, self.preprocess_clip = self.load_clip(clip_model_name)

        self.vit = nn.ModuleDict(
            {name: MotionViT(self.parts_output_dim[name], num_layers, num_heads) for name in self.parts_output_dim}
        )
        self.cls_token = {
            # name: nn.Parameter(torch.randn(1, 1, part_dim)) for name, part_dim in self.parts_output_dim.items()
            name: self.create_positional_encoding(part_dim, 1, idx) for idx, (name, part_dim) in enumerate(self.parts_output_dim.items())
        }
        # Masked text prediction
        self.mask_decoder = nn.ModuleDict(
            # {name: CrossAttentionDecoder(self.parts_output_dim[name], text_dim) for name in self.parts_output_dim}
            {name: CrossAttentionDecoder(d_model=512, vocab_size=self.clip.vocab_size) for name in self.parts_output_dim}
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
        # self.tau = nn.Parameter(torch.tensor(0.07))
        # Loss weights
        self.alpha = 0.5  # GSA weight
        self.beta = 0.3   # MTP weight
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
        model, preprocess = clip.load(model_name)
        # 冻结所有CLIP参数
        for param in model.parameters():
            param.requires_grad = False
        return model, preprocess
    
    def create_positional_encoding(self, dim, length, idx):
        """生成位置编码"""
        position = torch.arange(length).unsqueeze(1) + idx  # 为每个部位生成不同的编码
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(length, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    
    def _get_text_features(self, text):
        """获取文本特征"""
        text_tokens = self.clip.tokenize(text)
        # text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        # text_tokens = {k: v.to(self.clip.device) for k, v in text_tokens.items()}
        # text_features = self.clip.get_text_features(**text_tokens)
        text_features_allseq = self.clip.encode_text_allseq(text_tokens)
        feature = text_features_allseq[torch.arange(text_features_allseq.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip.text_projection
        return text_tokens, text_features_allseq.detach(), feature
    
    def forward(self, parts, text_tokens, text_feature, text_feature_all):
        """
        分身体前向传播
        :param parts: 身体部位特征列表 [Root, R_Leg, ...]
        :param text: 文本描述列表
        :return: 各模块输出
        """
        # # 文本特征处理
        # # text_tokens, text_feat = self._get_text_features(text)  # [B, text_dim]
        # if isinstance(text, tuple):
        #     # text = text[0]
        #     text = [name for id, name in enumerate(text)]
        # if isinstance(text, str):
        #     text = [text]
        # if isinstance(text, list):
        #     text_tokens, text_feat = self._get_text_features(text)
        # else:
        #     text_feat = text
        text_feature_all = text_feature_all.float()
        
        text_feat = text_feature.float()
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
            masked_feat = self._mask_text_feature(text_tokens)
            pred_logits = self.mask_decoder[name](z_vt, masked_feat)
            outputs['mtp_logits'].append((pred_logits.squeeze(1), text_feat))
            
            # 关系对齐
            word_emb = self.relation_proj_text(text_feature_all)  # [B, 256]
            code_emb = self.relation_proj_motion[name](z_vt[:, 1:].mean(dim=1))  # [B, 256]
            sim_matrix = F.cosine_similarity(word_emb.unsqueeze(1), code_emb.unsqueeze(0), dim=-1)
            outputs['relation_sims'].append(sim_matrix)
        
        return outputs

    def calculate_loss(self, outputs, text_ids):
        """计算多任务损失"""
        # 基础VQ损失
        # vq_loss = torch.stack(outputs['vq_losses']).mean()
        # perplexity = torch.stack(outputs['perplexity']).mean()
        # 全局语义对齐损失 (对比学习)
        gsa_loss = 0
        B = len(outputs['gsa_features'])
        e_TEXT_norm_list = []
        for e_CLS, e_TEXT in outputs['gsa_features']:
            e_TEXT_norm = F.normalize(e_TEXT, p=2, dim=1)
            e_TEXT_norm_list.append(e_TEXT_norm)
            
        # 构建text_id到样本索引的映射
        from collections import defaultdict
        id_to_indices = defaultdict(list)
        for idx, tid in enumerate(text_ids):
            id_to_indices[tid].append(idx)
        
        for i in range(B):
            e_CLS_i, e_TEXT_i = outputs['gsa_features'][i]
            e_CLS_norm_i = F.normalize(e_CLS_i, p=2, dim=1)
            
            # 计算当前样本与其他所有样本的相似度矩阵行
            similarity_matrix_i = torch.matmul(
                e_CLS_norm_i,
                e_TEXT_norm_list[i].t()  # [B, D] x [B, D] => [B, B]
            )
            probs_i = torch.sigmoid(similarity_matrix_i)
            log_probs_i = -torch.log(probs_i + 1e-12)  # [B, B]
            
            current_tid = text_ids[i]
            same_class_indices = id_to_indices[current_tid]
            
            # 定义正样本：除自己外的同类样本
            pos_indices = [j for j in same_class_indices if j != i]
            # 定义负样本：不同类的样本
            neg_indices = [j for j in range(B) if j not in same_class_indices]
            
            # 计算正样本平均log_prob
            avg_pos = 0.0
            if pos_indices:
                avg_pos = log_probs_i[i, pos_indices].mean()
            
            # 计算负样本平均log_prob
            avg_neg = 0.0
            if neg_indices:
                avg_neg = log_probs_i[i, neg_indices].mean()
            
            # 损失项：正样本对数概率 - 负样本对数概率(越高越好)
            loss_i = avg_pos - avg_neg
            gsa_loss -= loss_i
        
        # gsa_loss /= B
        # for e_CLS, e_TEXT in outputs['gsa_features']:
        #     # # 添加温度缩放
        #     # logits = torch.matmul(
        #     #     F.normalize(e_CLS, p=2, dim=1),
        #     #     F.normalize(e_TEXT, p=2, dim=1).t()
        #     # ) #/ self.tau.clamp(min=1e-4)
        #     # labels = torch.arange(logits.size(0), device=logits.device)
        #     # gsa_loss += F.cross_entropy(logits, labels)
        #     # 添加温度缩放和特征归一化
        #     e_CLS_norm = F.normalize(e_CLS, p=2, dim=1)  # [B, D]
        #     e_TEXT_norm = F.normalize(e_TEXT, p=2, dim=1)  # [B, text_dim]
            
        #     # 计算相似度矩阵
        #     similarity_matrix = torch.matmul(e_CLS_norm, e_TEXT_norm.t())  # [B, B]
            
        #     # 应用温度缩放
        #     # scaled_similarity = similarity_matrix / self.tau.clamp(min=1e-4, max=1.0)  # [B, B]
            
        #     # 通过sigmoid转换为概率
        #     probs = torch.sigmoid(similarity_matrix)  # [B, B]
            
        #     # InfoNCE损失计算
        #     # 正样本是自身位置，负样本是其他所有位置
        #     log_probs = -torch.log(probs + 1e-12)  # 防止log(0)
            
        #     # 每个样本的损失：取自己位置的正样本log_prob，其他为负样本
        #     loss_i = log_probs[range(len(probs)), range(len(probs))] - log_probs.sum(dim=1)
        #     gsa_loss -= loss_i.mean()
            
        
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

    def _mask_text_feature(self, text_tokens, mask_ratio=0.15):
        """生成掩码文本特征"""
        # masks = []
        # B, L = text_tokens.size()
        masks = torch.ones_like(text_tokens, dtype=torch.bool)  # 初始化为全True的掩码
        for i in range(text_tokens.shape[0]):
            text_token = text_tokens[i][:torch.argmax(text_tokens[i], dim=-1)]
            num_mask = int(mask_ratio * (text_token.shape[0]-2))
            mask_idx = torch.randperm(text_token.shape[0]-2)[:num_mask] + 1
            # masks.append(mask_idx.clone())
            masks[i, mask_idx] = False  # 将选中的位置设置为False

        masked_text_tokens = text_tokens * masks  # 应用掩码
        return masked_text_tokens  

def plot_tsne(e_CLS, e_TEXT):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    # import numpy as np
    tsne = TSNE(n_components=2)
    vis_feat = tsne.fit_transform(torch.cat([e_CLS.cpu().detach(), e_TEXT.cpu().detach()]))
    labels = torch.cat([torch.ones(e_CLS.cpu().detach().size(0)), torch.ones(e_TEXT.cpu().detach().size(0))*2])
    plt.scatter(vis_feat[:,0], vis_feat[:,1], c=labels)
    plt.savefig("CLS_TEXT2.png")
    
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

# class CrossModalCLS(nn.Module):
#     def __init__(self, part_dim, text_dim, num_heads=4):
#         super().__init__()
#         # 跨模态注意力机制
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=part_dim,
#             kdim=text_dim,
#             vdim=text_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         # 动态生成层
#         self.generator = nn.Sequential(
#             nn.Linear(part_dim, 4*part_dim),
#             nn.LayerNorm(4*part_dim),
#             nn.GELU(),
#             nn.Linear(4*part_dim, part_dim)
#         )
        
#     def forward(self, part_feat, text_feat):
#         """
#         part_feat: [B, T, D] 部位运动特征
#         text_feat: [B, S, text_dim] 文本特征序列
#         """
#         B = part_feat.size(0)
        
#         # 初始化查询向量
#         query = torch.mean(part_feat, dim=1, keepdim=True)  # [B, 1, D]
        
#         # 跨模态注意力
#         cls_token, _ = self.cross_attn(
#             query=query,
#             key=text_feat,
#             value=text_feat
#         )  # [B, 1, D]
        
#         # 动态增强
#         cls_token = self.generator(cls_token)
        
#         return torch.cat([cls_token, part_feat], dim=1)

class CrossAttentionDecoder(torch.nn.Module):
    def __init__(self, d_model=512, nhead=8, vocab_size=49408):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = torch.nn.Linear(d_model, vocab_size)
        
    def forward(self, visual_codes, text_embeddings):
        # visual_codes: [B, L_v, d_z], text_embeddings: [B, L_t, d_t]
        attn_output, _ = self.cross_attn(
            query=text_embeddings, 
            key=visual_codes,
            value=visual_codes
        )
        logits = self.ffn(attn_output)
        return logits  # [B, L_t, vocab_size]

# class CrossAttentionDecoder(nn.Module):
#     def __init__(self, motion_dim, text_dim=512):  # CLIP默认文本维度
#         super().__init__()
        
#         # 维度对齐投影层
#         self.motion_proj = nn.Linear(motion_dim, text_dim)
        
#         # 跨注意力机制（使用文本维度）
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=text_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         # 前馈网络
#         self.ffn = nn.Sequential(
#             nn.Linear(text_dim, 4*text_dim),
#             nn.GELU(),
#             nn.Linear(4*text_dim, text_dim)
#         )
        
#         # 输出投影层（适配CLIP特征空间）
#         self.proj = nn.Linear(text_dim, text_dim)  # 保持维度一致

#     def forward(self, motion_feat, masked_text_feat):
#         """
#         motion_feat: [B, T+1, motion_dim] 运动特征（含CLS token）
#         masked_text_feat: [B, S, text_dim] 被掩码的CLIP文本特征
#         """
#         # 运动特征维度投影
#         projected_motion = self.motion_proj(motion_feat)  # [B, T+1, text_dim]
        
#         # 跨注意力计算（Query: 文本， Key/Value: 运动）
#         attn_out, _ = self.cross_attn(
#             query=masked_text_feat,
#             key=projected_motion,
#             value=projected_motion
#         )
        
#         # 特征增强
#         ffn_out = self.ffn(attn_out)
        
#         # 最终投影（保持与CLIP特征相同维度）
#         return self.proj(ffn_out)  # [B, S, text_dim]
    