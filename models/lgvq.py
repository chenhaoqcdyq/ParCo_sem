import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip
from models.quantize_cnn import QuantizeEMAReset
# from models.rvqvae_bodypart import CausalTransformerEncoder, ContrastiveLossWithSTSV2
from models.vqvae_bodypart import VQVAE_bodypart
from models.encdec import RepeatFirstElementPad1d


class CausalTransformerEncoder(nn.TransformerEncoder):
    """带因果掩码的Transformer编码器"""
    def forward(self, src, mask=None, **kwargs):
        # 自动生成因果掩码
        if mask is None:
            device = src.device
            seq_len = src.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            return super().forward(src, mask=causal_mask, **kwargs)
        return super().forward(src, mask, **kwargs)

class ContrastiveLossWithSTS(nn.Module):
    def __init__(self, temperature=0.07, threshold=0.85):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

    def _get_similarity_matrix(self, text_embeds):
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        sim_matrix = torch.mm(text_embeds, text_embeds.T)
        return sim_matrix.cpu()

    def forward(self, motion_feat, text_feat, texts_feature):
        """
        motion_feat: [B, D] 动作特征
        text_feat: [B, D] 文本特征
        texts: List[str] 原始文本
        """
        # 计算文本语义相似度
        sim_matrix = self._get_similarity_matrix(texts_feature)  # [B, B]
        
        # 生成正样本掩码（GPU计算）
        pos_mask = (sim_matrix > self.threshold).float().to(motion_feat.device)
        # pos_mask.fill_diagonal_(0)  # 排除自身
        
        # 特征归一化
        motion_feat = F.normalize(motion_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # 计算logits
        logits_per_motion = torch.mm(motion_feat, text_feat.T) / self.temperature  # [B, B]
        logits_per_text = logits_per_motion.T
        
        # 多正样本对比损失
        exp_logits = torch.exp(logits_per_motion)
        numerator = torch.sum(exp_logits * pos_mask, dim=1)  # 分子：正样本相似度
        denominator = torch.sum(exp_logits, dim=1)          # 分母：所有样本
        
        # 避免除零
        valid_pos = (pos_mask.sum(dim=1) > 0)
        loss_motion = -torch.log(numerator[valid_pos]/denominator[valid_pos]).sum()
        
        # 对称文本到动作损失
        exp_logits_text = torch.exp(logits_per_text)
        numerator_text = torch.sum(exp_logits_text * pos_mask, dim=1)
        denominator_text = torch.sum(exp_logits_text, dim=1)
        loss_text = -torch.log(numerator_text[valid_pos]/denominator_text[valid_pos]).sum()
        
        return (loss_motion + loss_text) / 2

class ContrastiveLossWithSTSV2(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_feat, text_feat, text_id):
        """
        motion_feat: [B, D] 动作特征
        text_feat: [B, D] 文本特征
        texts: List[str] 原始文本
        """
        pos_mask = torch.zeros(len(text_id), len(text_id), device=motion_feat.device)
        for i in range(len(text_id)):
            for j in range(len(text_id)):
                if text_id[i] == text_id[j]:
                    pos_mask[i, j] = 1
        # 特征归一化
        motion_feat = F.normalize(motion_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # 计算logits
        logits_per_motion = torch.mm(motion_feat, text_feat.T) / self.temperature  # [B, B]
        logits_per_text = logits_per_motion.T
        
        # 多正样本对比损失
        exp_logits = torch.exp(logits_per_motion)
        numerator = torch.sum(exp_logits * pos_mask, dim=1)  # 分子：正样本相似度
        denominator = torch.sum(exp_logits, dim=1)          # 分母：所有样本
        
        # 避免除零
        valid_pos = (pos_mask.sum(dim=1) > 0)
        loss_motion = -torch.log(numerator[valid_pos]/denominator[valid_pos]).mean()
        
        # 对称文本到动作损失
        exp_logits_text = torch.exp(logits_per_text)
        numerator_text = torch.sum(exp_logits_text * pos_mask, dim=1)
        denominator_text = torch.sum(exp_logits_text, dim=1)
        loss_text = -torch.log(numerator_text[valid_pos]/denominator_text[valid_pos]).mean()
        
        return (loss_motion + loss_text) / 2
    
    def compute_disentangle_loss(self, quant_vis, quant_sem, disentanglement_ratio=0.1):
        quant_vis = rearrange(quant_vis, 'b t c -> (b t) c')
        quant_sem = rearrange(quant_sem, 'b t c -> (b t) c')

        quant_vis = F.normalize(quant_vis, p=2, dim=-1)
        quant_sem = F.normalize(quant_sem, p=2, dim=-1)

        dot_product = torch.sum(quant_vis * quant_sem, dim=1)
        loss = torch.mean(dot_product ** 2) * disentanglement_ratio

        return loss

# clip版本的lgvq
class LGVQ(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 global_token_mode = 1,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 0 采用vit clstoken， 1采用pooling
        self.global_token_mode = global_token_mode
        if global_token_mode == 0:
            self.global_time_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
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
        # args
        self.args = args
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.text_proj = nn.Linear(args.text_dim, d_model)
        self.motion_text_proj = nn.Linear(d_model, d_model)

    def forward(self, parts_feature, text=None):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i,  feat in enumerate(parts_feature):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        if self.global_token_mode == 0:
            global_time_tokens = self.global_time_token.expand(B*7, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # spatial_cube = rearrange(spatial_cube, 'b t p d -> (b p) t d')
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if self.global_token_mode==0:
            time_feat = torch.cat([
                global_time_tokens,
                time_feat
            ], dim=1)
        time_feat = self.time_transformer(time_feat)  # [B*7, T, d]
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        if self.global_token_mode == 0:
            global_feat = feature[:, 0, 0, :]
        else:
            global_feat = feature[:, :, 0, :].mean(dim=1)
        if text is not None:
            text_feature, text_id = text
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_feature = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)  # [B, d_model]
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature, contrastive_loss
    
from transformers import BertTokenizer, BertModel
class LGVQv2(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 global_token_mode = 1,
                 bert_hidden_dim = 768,
                 vocab_size = 30522,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 0 采用vit clstoken， 1采用pooling
        self.global_token_mode = global_token_mode
        if global_token_mode == 0:
            self.global_time_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
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
        # args
        self.args = args
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.text_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 增强跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim,
            num_heads=8,
            batch_first=True
        )
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(args.d_model * 7, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU()
        )
        
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        self.temporal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bert_hidden_dim,
                nhead=8,
                dim_feedforward=3072,
                batch_first=True),
            num_layers=2
        )
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_model.cuda()
        self.bert_model.eval()  # 冻结BERT参数
        self.vocab_size = vocab_size
        self.motion_text_proj = nn.Linear(d_model, bert_hidden_dim)

    def forward(self, parts_feature, text=None, text_mask = None, motion_mask=None):
        self.bert_model.to(parts_feature[0].device)
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i,  feat in enumerate(parts_feature):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        if self.global_token_mode == 0:
            global_time_tokens = self.global_time_token.expand(B*7, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if self.global_token_mode==0:
            time_feat = torch.cat([
                global_time_tokens,
                time_feat
            ], dim=1)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [T, B*7, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        if self.global_token_mode == 0:
            global_feat = feature[:, 0, 0, :]
        else:
            global_feat = feature[:, :, 0, :].mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                # text_feature = text_mask['bert_features']
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)  # [B, seq_len]
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
            text_feature = text_feature.to(parts_feature[0].device).float()
            # text_feature = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)  # [B, d_model]
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            # 跨模态注意力
            if motion_mask is not None:
                attn_output, _ = self.cross_attention(
                    query=text_feature,
                    key=motion_query,
                    value=motion_query,
                    key_padding_mask=~motion_mask
                )  # [bs, seq_text, bert_hidden]
            else:
                attn_output, _ = self.cross_attention(
                    query=text_feature,
                    key=motion_query,
                    value=motion_query
                )
            fused_features = self.temporal_fusion(
                attn_output,
                src_key_padding_mask=~attention_mask
            )  # [bs, seq_text, bert_hidden]
            logits = self.mlm_head(fused_features)  # [bs, seq_text, vocab_size]
            
            # 计算掩码损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            mlm_loss = loss_fct(active_logits, active_labels.long())
            text_feature_pooler = self.text_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature, [contrastive_loss, mlm_loss]
    
class LGVQv3(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 bert_hidden_dim = 768,
                 vocab_size = 30522,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
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
        # args
        self.args = args
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.text_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.query_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        # 增强跨模态注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=8,
                dim_feedforward=4*bert_hidden_dim,
                batch_first=True
            ) for _ in range(3)
        ])
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(args.d_model * 7, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU()
        )
        
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_model.cuda()
        self.bert_model.eval()  # 冻结BERT参数
        self.vocab_size = vocab_size
        self.motion_text_proj = nn.Linear(d_model, bert_hidden_dim)
        
    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        # 部件特征预处理
        B, T = motion[0].shape[0], motion[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i,  feat in enumerate(motion):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [T, B*7, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)
        
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_proj(text_feature)
        
        # similarity = F.cosine_similarity(motion_feature_global, text_feature_pooler, dim=-1)
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        if text_mask is not None:
            # text_feature = text_mask['bert_features']
            input_ids = text_mask['input_ids'].to(motion[0].device)  # [B, seq_len]
            labels = text_mask['labels'].to(motion[0].device).float()
            attention_mask = text_mask['attention_mask'].to(motion[0].device).bool()
            # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion[0].device).float()
                text_feature_pooler = text_mask['feature'].to(motion[0].device).float()
            text_query = self.query_proj(text_feature)
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            logits = self.mlm_head(text_query)  # [bs, seq_text, vocab_size]
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
    
            # 将labels转为整数索引
            active_labels = active_labels.long()  # [active_num]
            
            # 计算是否命中
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            # 统计R1/R3/R5
            r1 = hits[:, 0].sum().float() / active_labels.size(0)
            r3 = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5 = hits.sum().float() / active_labels.size(0)
            return [correct_r1/ batch_size, correct_r3/ batch_size, correct_r5/ batch_size] , [r1.cpu().item(), r3.cpu().item(), r5.cpu().item()]
        return [correct_r1/ batch_size, correct_r3/ batch_size, correct_r5/ batch_size], [0,0,0]

    def forward(self, parts_feature, text=None, text_mask = None, motion_mask=None):
        # self.bert_model.to(parts_feature[0].device)
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i,  feat in enumerate(parts_feature):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [T, B*7, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                # text_feature = text_mask['bert_features']
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)  # [B, seq_len]
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_query = self.query_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)  # [B, d_model]
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            # attn_mask = self._generate_cross_mask(attention_mask, motion_mask)
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            logits = self.mlm_head(text_query)  # [bs, seq_text, vocab_size]
            
            # 计算掩码损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            mlm_loss = loss_fct(active_logits, active_labels.long())
            text_feature_pooler = self.text_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature, [contrastive_loss, mlm_loss]

class LGVQv4(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=8,
                 num_layers=2,
                 bert_hidden_dim = 768,
                 vocab_size = 30522,
                 down_sample = False,
                 ):
        super().__init__()
        # 全身特征聚合Token
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 增强的位置编码体系
        self.part_position = nn.Embedding(6, d_model)  # 部件类型编码
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
        self.time_transformer = nn.TransformerEncoder(time_encoder_layer, num_layers=num_layers)
        # args
        self.args = args
        # self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.contrastive_loss_v2 = ContrastiveLossWithSTSV3()
        self.text_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.query_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        # 增强跨模态注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=8,
                dim_feedforward=4*bert_hidden_dim,
                batch_first=True,
                dropout=0.1
            ) for _ in range(2)
        ])
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(args.d_model * 7, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU()
        )
        
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_model.cuda()
        self.bert_model.eval()  # 冻结BERT参数
        self.vocab_size = vocab_size
        self.motion_text_proj = nn.Linear(d_model, bert_hidden_dim)
        self.symmetric_mask = SymmetricPartMask()
        self.spatio_temporal_cutmix = SpatioTemporalCutMix(12, 24)
        self.part_mix = PartMix()
        
    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        # 部件特征预处理
        B, T = motion[0].shape[0], motion[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(motion):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        # spatial_cube = self.down_sample(spatial_cube)
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)  # [B*7, T, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)
        
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_proj(text_feature)
        
        # similarity = F.cosine_similarity(motion_feature_global, text_feature_pooler, dim=-1)
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        if text_mask is not None:
            # text_feature = text_mask['bert_features']
            input_ids = text_mask['input_ids'].to(motion[0].device)  # [B, seq_len]
            labels = text_mask['labels'].to(motion[0].device).float()
            attention_mask = text_mask['attention_mask'].to(motion[0].device).bool()
            # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion[0].device).float()
                text_feature_pooler = text_mask['feature'].to(motion[0].device).float()
            text_query = self.query_proj(text_feature)
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            logits = self.mlm_head(text_query)  # [bs, seq_text, vocab_size]
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
    
            # 将labels转为整数索引
            active_labels = active_labels.long()  # [active_num]
            
            # 计算是否命中
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            # 统计R1/R3/R5
            r1 = hits[:, 0].sum().float() / active_labels.size(0)
            r3 = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5 = hits.sum().float() / active_labels.size(0)
            return [correct_r1/ batch_size, correct_r3/ batch_size, correct_r5/ batch_size] , [r1.cpu().item(), r3.cpu().item(), r5.cpu().item()]
        return [correct_r1/ batch_size, correct_r3/ batch_size, correct_r5/ batch_size], [0,0,0]

    def forward(self, parts_feature, text=None, text_mask = None, motion_mask=None):
        # 部件特征预处理
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        # 时空位置编码注入
        part_embeds = []
        for i,  feat in enumerate(parts_feature):
            # 部件类型编码
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        if motion_mask is not None:
            motion_mask = motion_mask.to(parts_feature[0].device).bool()
            time_mask_expanded = motion_mask.unsqueeze(-1).expand(-1, -1, 7).bool().to(parts_feature[0].device)
        # 构建时空特征立方体 [B, T, 6, d_model]
        spatial_cube = torch.stack(part_embeds, dim=2)
        spatial_cube, motion_mask = self.spatio_temporal_cutmix(spatial_cube, motion_mask)
        text_feature, text_id = text
        spatial_cube, text_id_new = self.part_mix(spatial_cube, text_id)
        # spatial_cube, spatial_mask = self.symmetric_mask(spatial_cube, 2)
        
        # 构建融合输入 [B*T, 7, d_model]（6个部件+1个全局Token）
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        # all_mask = create_spatial_padding_mask(motion_mask, spatial_mask)
        # 空间维度交互 (部件间关系)        spatial_cube = spatial_cube + self.time_position[:, :T, :][None, :, None, :]
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        
        spatial_feat = self.spatial_transformer(spatial_feat)  # [B*T, 7, d]
        
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            # time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=rearrange(~time_mask_expanded, 'b t p-> (b p) t'))  # [B*7, T, d]
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                # text_feature = text_mask['bert_features']
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)  # [B, seq_len]
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_query = self.query_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)  # [B, d_model]
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            # attn_mask = self._generate_cross_mask(attention_mask, motion_mask)
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            logits = self.mlm_head(text_query)  # [bs, seq_text, vocab_size]
            
            # 计算掩码损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            mlm_loss = loss_fct(active_logits, active_labels.long())
            text_feature_pooler = self.text_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss_v2(motion_feature_global, text_feature_pooler, text_id_new)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature, [contrastive_loss, mlm_loss]

class PartMix(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, parts_feature, labels):
        """
        输入:
            parts_feature: [B, T, 6, D] 或 list([B, T, D] * 6)
            labels: [B,] 用于对比学习的文本ID
        返回:
            混合特征和调整后的标签
        """
        if not self.training or self.alpha <= 0:
            return parts_feature, labels
            
        # 生成混合比例
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(parts_feature.device)
        
        # 随机选择样本对
        B = parts_feature.size(0)
        index = torch.randperm(B).to(parts_feature.device)
        
        # 应用混合
        mixed_feat = parts_feature.clone()
        # 随机选择要混合的部件
        part_idx = torch.randperm(4, device=parts_feature.device)[:2] + 1  # 不混合Root/Backbone
        mixed_feat[:, :, part_idx] = lam.unsqueeze(0).unsqueeze(1).unsqueeze(2) * parts_feature[:, :, part_idx] + \
                                    (1 - lam).unsqueeze(0).unsqueeze(1).unsqueeze(2) * parts_feature[index][:, :, part_idx]
        
        # 标签调整 - 处理字符串列表
        if isinstance(labels, list) or isinstance(labels, tuple):
            # 如果是字符串列表，创建新的混合标签列表
            orig_labels = labels
            mix_labels = [labels[i] for i in index.cpu().numpy()]
            lam_tensor = torch.full((B,), lam, device=parts_feature.device)
            new_labels = (orig_labels, mix_labels, lam_tensor)
        else:
            # 如果是张量，直接使用索引
            new_labels = (labels, labels[index], torch.full((B,), lam, device=parts_feature.device))
        
        return mixed_feat, new_labels

class ContrastiveLossWithSTSV3(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_feat, text_feat, labels):
        """
        输入:
            motion_feat: [B, D] 动作特征 
            text_feat: [B, D] 文本特征
            labels: (orig_labels, mix_labels, lam)
                orig_labels: [B,] 原始文本字符串
                mix_labels: [B,] 混合文本字符串
                lam: [B,] 混合系数
        """
        orig_label, mix_label, lam = labels
        B = motion_feat.size(0)
        
        # 特征归一化
        motion_feat = F.normalize(motion_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # 计算相似度矩阵
        logits_per_motion = torch.mm(motion_feat, text_feat.T) / self.temperature  # [B, B]
        logits_per_text = logits_per_motion.T
        
        # 构造动态正样本掩码
        def build_pos_mask(src_label, tgt_label, mix_src_label, lam):
            # 源标签与目标标签匹配 (原始混合)
            pos_orig = torch.tensor([[1 if s == t else 0 for t in tgt_label] for s in src_label], 
                                  device=lam.device).float() * lam.unsqueeze(1)
            # 源混合标签与目标标签匹配 (混合部分)
            pos_mix = torch.tensor([[1 if s == t else 0 for t in tgt_label] for s in mix_src_label], 
                                 device=lam.device).float() * (1 - lam).unsqueeze(1)
            return pos_orig + pos_mix
        
        # 动作->文本正样本掩码
        pos_mask_motion = build_pos_mask(orig_label, orig_label, mix_label, lam)
        
        # 文本->动作正样本掩码 (需交换角色)
        pos_mask_text = build_pos_mask(orig_label, orig_label, mix_label, lam).T
        
        # 双向对比损失计算
        def contrastive_loss(logits, pos_mask):
            exp_logits = torch.exp(logits)
            numerator = torch.sum(exp_logits * pos_mask, dim=1)
            denominator = torch.sum(exp_logits, dim=1) + 1e-8
            return -torch.log(numerator / denominator).mean()
        
        loss_motion = contrastive_loss(logits_per_motion, pos_mask_motion)
        loss_text = contrastive_loss(logits_per_text, pos_mask_text)
        
        return (loss_motion + loss_text) / 2

class SpatioTemporalCutMix(nn.Module):
    def __init__(self, min_t=8, max_t=16):
        super().__init__()
        self.min_t = min_t
        self.max_t = max_t
        
    def forward(self, parts_feature, motion_mask = None):
        """
        输入:
            parts_feature: [B, T, 6, D] (已堆叠的部件特征)
            motion_mask: [B, T] 时间掩码 (True表示有效位置)
        返回:
            混合后的特征和调整后的mask
        """
        if not self.training:
            return parts_feature, motion_mask
            
        B, T, P, D = parts_feature.shape
        device = parts_feature.device
        
        # 随机选择时间窗口
        t_len = torch.randint(self.min_t, self.max_t+1, (1,), device=device).item()
        t_start = torch.randint(0, T - t_len, (1,), device=device).item()
        
        # 随机选择部件
        part_idx = torch.randperm(4)[:2].add(1).to(device)  # 不混合Root/Backbone
        
        # 生成样本对
        index = torch.randperm(B).to(device)
        
        # 应用CutMix
        mixed_feat = parts_feature.clone()
        mixed_feat[:, t_start:t_start+t_len, part_idx] = parts_feature[index][:, t_start:t_start+t_len, part_idx]
        
        # 调整时间掩码 - 确保混合区域的时间掩码也正确混合
        mixed_mask = motion_mask.clone()
        if motion_mask is not None: 
            # 在混合区域，如果任一样本在该时间点有效，则保留该时间点
            mixed_mask[:, t_start:t_start+t_len] = motion_mask[:, t_start:t_start+t_len] | motion_mask[index][:, t_start:t_start+t_len]
        else:
            mixed_mask = None
        return mixed_feat, mixed_mask

class SymmetricPartMask(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask_num=0):
        # x: [B, T, 6, D] (部件顺序需与part_names一致)
        if not self.training or mask_num == 0:
            return x
        B, T, P, D = x.shape
        mask = torch.ones(B, T, P).bool().to(x.device)
        for b in range(B):
            for t in range(T):
                selected = random.sample(range(P), mask_num)
                mask[b,t,selected] = True
        # 保留Root和Backbone的完整性
        mask[:, :, [0, 3]] = False
        x_masked = x * (~mask.unsqueeze(-1)).float()
        return x_masked, mask
    
class SymmetricPartMaskV2(nn.Module):
    def __init__(self, 
                 part_names=['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm'],
                 symmetric_pairs=[('R_Leg', 'L_Leg'), ('R_Arm', 'L_Arm')],
                 max_mask=2):
        super().__init__()
        # 构建索引映射
        self.part_idx = {name:i for i,name in enumerate(part_names)}
        self.mask_pairs = [
            (self.part_idx[p1], self.part_idx[p2]) 
            for p1,p2 in symmetric_pairs
        ]
        self.max_mask = max_mask
        
    def forward(self, x):
        # x: [B, T, 6, D] (部件顺序需与part_names一致)
        if not self.training:
            return x
        
        # 随机选择掩码对
        mask_num = random.sample(range(min(self.max_mask, len(self.mask_pairs))))
        selected = random.sample(self.mask_pairs, mask_num)
        
        # 生成掩码
        mask = torch.ones_like(x)
        for (idx1, idx2) in selected:
            mask[:, :, [idx1, idx2], :] = 0
            
        # 保留Root和Backbone的完整性
        mask[:, :, [0, 3], :] = 1  # 确保Root和Backbone不被掩码
        return x * mask

def create_spatial_padding_mask(motion_mask, spatial_mask):
    """
    motion_mask: [B, T] 时间维度padding掩码 (True表示有效位置)
    spatial_mask: [B, T, 6] 部件掩码 (False表示有效位置)
    返回: [B*T, 7] 合并后的padding掩码 (False表示需要mask的位置)
    """
    B, T = motion_mask.shape
    # 扩展时间掩码到7个位置（6部件+全局token）
    time_mask_expanded = motion_mask.unsqueeze(-1).expand(-1, -1, 7).bool().to(spatial_mask.device) # [B, T, 7]
    
    # 扩展部件掩码（添加全局token位置为True）
    spatial_mask_expanded = torch.cat([
        torch.zeros(B, T, 1).bool().to(spatial_mask.device),  # 全局token不mask
        spatial_mask
    ], dim=2)  # [B, T, 7]
    
    # 合并掩码：仅当时间有效且部件有效时才保留
    combined_mask = ~time_mask_expanded & spatial_mask_expanded
    
    # 重组为Transformer输入格式
    return combined_mask

class LGVQv5(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=4,  # 减少注意力头数
                 num_layers=1,  # 减少Transformer层数
                 bert_hidden_dim=768,
                 vocab_size=30522,
                 dropout=0.2,   # 增加dropout率
                 down_sample = False):  
        super().__init__()
        # 改进1: 简化结构 + 正则化
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.part_position = nn.Embedding(6, d_model)
        self.vocab_size = vocab_size
        # 改进2: 增加Transformer层的Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,  # 增加dropout
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 改进3: 增强时间建模正则化
        self.time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2*d_model,  # 减少FFN维度
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 改进4: 部分微调BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # 默认冻结


        # 改进5: 加强投影层正则化
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.text_motion_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 改进6: 增强跨模态注意力正则化
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=4,  # 减少注意力头
                dim_feedforward=2*bert_hidden_dim,  # 降低FFN维度
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 减少层数
        ])

        # 改进7: 对比学习增强
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        
        # 改进8: 数据增强
        self.motion_aug = ComposeAugmentation([
            # TemporalCrop(max_ratio=0.2),
            FeatureJitter(std=0.05)
        ])
        
        # 改进9: 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(d_model * 7, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        # 改进10: MLM head with label smoothing
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        
        # 改进11: 运动文本投影
        self.motion_text_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.ifdown_sample = down_sample
        if down_sample:
            self.down_sample = TemporalDownsamplerV3(d_model)
        else:
            self.down_sample = nn.Identity()

    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        """
        计算动作和文本之间的Top-K匹配
        Args:
            motion: 动作特征列表 [6, B, T, D]
            text: 文本字符串
            motion_mask: 动作掩码 [B, T]
            topk: 返回的top-k结果数
            text_mask: 文本掩码字典
        Returns:
            [r1, r3, r5]: 召回率指标
            [r1_mlm, r3_mlm, r5_mlm]: MLM任务的召回率指标
        """
        # 部件特征预处理
        B, T = motion[0].shape[0], motion[0].shape[1]
        if self.ifdown_sample:
            T = T // 4
            
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(motion):
            part_embeds.append(self.down_sample(feat + self.part_position.weight[i][None, None, :]))
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        
            
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            time_feat = self.time_transformer(time_feat)
            
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)
        
        # 文本特征提取
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_motion_proj(text_feature)
        
        # 计算相似度矩阵
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        # MLM任务的召回率计算
        if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion[0].device)
            labels = text_mask['labels'].to(motion[0].device).float()
            attention_mask = text_mask['attention_mask'].to(motion[0].device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion[0].device).float()
                text_feature_pooler = text_mask['feature'].to(motion[0].device).float()
                
            # 特征投影
            text_query = self.text_proj(text_feature)
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 计算MLM任务的Top-K召回率
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
            active_labels = active_labels.long()  # [active_num]
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            r1_mlm = hits[:, 0].sum().float() / active_labels.size(0)
            r3_mlm = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5_mlm = hits.sum().float() / active_labels.size(0)
            
            return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], \
                   [r1_mlm.cpu().item(), r3_mlm.cpu().item(), r5_mlm.cpu().item()]
                   
        return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], [0, 0, 0]

    def forward(self, parts_feature, text=None, text_mask=None, motion_mask=None):
        # 部件特征预处理 bs,6,seq,d
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        if self.ifdown_sample:
            T = T // 4
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(parts_feature):
            part_embeds.append(self.down_sample(feat + self.part_position.weight[i][None, None, :]))
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        # spatial_cube = self.down_sample(spatial_cube)
        # 数据增强
        if self.training:
            spatial_cube = self.motion_aug(spatial_cube)
            
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            time_feat = self.time_transformer(time_feat)
            
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        global_feat = feature[:, :, 0, :].mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
                    
            # 特征投影
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(rearrange(feature, 'b t p d -> b t (p d)'))
            
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
            mlm_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature, [contrastive_loss, mlm_loss]

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ComposeAugmentation(nn.Module):
    """组合时空数据增强"""
    def __init__(self, aug_list):
        super().__init__()
        self.aug_list = aug_list

    def forward(self, x):
        for aug in self.aug_list:
            x = aug(x)
        return x

class TemporalCrop(nn.Module):
    """时间维度随机裁剪"""
    def __init__(self, max_ratio=0.2):
        super().__init__()
        self.max_ratio = max_ratio
        
    def forward(self, x):
        T = x.size(1)
        crop_len = int(T * torch.rand(1) * self.max_ratio)
        start = torch.randint(0, T-crop_len, (1,))
        return x[:, start:start+crop_len]

class FeatureJitter(nn.Module):
    """特征加噪"""
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class TemporalDownsamplerV3(nn.Module):
    """时间维度1/4降采样模块"""
    def __init__(self, d_model, causal=False):
        super().__init__()
        if causal:
            self.conv_layers = nn.Sequential(
                RepeatFirstElementPad1d(padding=2),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0),
                nn.GELU(),
                RepeatFirstElementPad1d(padding=2),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0)
            )
        else:
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

class TemporalDownsamplerHalf(nn.Module):
    """时间维度1/2降采样模块，使用单层卷积实现"""
    def __init__(self, d_model, causal=False):
        super().__init__()
        if causal:
            self.conv_layers = nn.Sequential(
                RepeatFirstElementPad1d(padding=2),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0),
                nn.GELU()
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU()
            )
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//2, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, T//2, C]
        return x

class Dualsem_encoder(nn.Module):
    def __init__(self, args, num_layers=4, d_model=256, nhead=8, bert_hidden_dim=768, vocab_size=30522):
        super().__init__()
        self.args = args
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        self.motion_text_proj = nn.Linear(d_model, args.text_dim)
        self.text_proj = nn.Linear(args.text_dim, args.text_dim)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        self.text_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 增强跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim,
            num_heads=8,
            batch_first=True
        )
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(args.d_model * 7, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        self.temporal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bert_hidden_dim,
                nhead=8,
                dim_feedforward=3072,
                batch_first=True),
            num_layers=2
        )
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()  # 冻结BERT参数
        self.vocab_size = vocab_size
        self.motion_text_proj = nn.Linear(d_model, bert_hidden_dim)
        self.sem_quantizer = QuantizeEMAReset(args.vqvae_sem_nb, d_model, args)
        for idx, name in enumerate(self.parts_name):
            quantizer = QuantizeEMAReset(args.vqvae_arch_cfg['parts_code_nb'][name], d_model, args)
            setattr(self, f'quantizer_{name}', quantizer)
        
    def forward(self, parts_feature, text=None, text_mask = None, motion_mask=None):
        self.bert_model.to(parts_feature[0].device)
        # 部件特征预处理
        P, B, T = parts_feature.shape[0], parts_feature.shape[1], parts_feature.shape[2]
        # 构建时空特征立方体 [B, T, 6, d_model]
        # time_feat = torch.cat(parts_feature, dim=0)
        # time_feat = rearrange(parts_feature, '(b p) t d-> (b p) t d', b=B, p=7)
        sptial_feat = rearrange(parts_feature, 'p b t d-> (b t) p d')
        sptial_feat = self.spatial_transformer(sptial_feat)
        time_feat = rearrange(sptial_feat, '(b t) p d-> (b p) t d', b=B)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            time_feat = self.time_transformer(time_feat)
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        # quantized
        x_quantized_sem, loss_sem, perplexity_sem = self.sem_quantizer(rearrange(feature[0], 'b t d -> b d t'))
        x_quantized_list = [x_quantized_sem.permute(0,2,1)]
        loss_list = [loss_sem]
        perplexity_list = [perplexity_sem]
        for idx, name in enumerate(self.parts_name):
            quantizer = getattr(self, f'quantizer_{name}')
            x_quantized, loss, perplexity = quantizer(rearrange(feature[idx+1], 'b t d -> b d t'))
            x_quantized_list.append(x_quantized.permute(0,2,1))
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            
        global_feat = x_quantized_sem.permute(0,2,1).mean(dim=1)
        feature_quantized = torch.cat(x_quantized_list, dim=2)
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                # text_feature = text_mask['bert_features']
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)  # [B, seq_len]
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                # text_feature_pooler = text_mask['bert_features_pool'].to(parts_feature[0].device).float()
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
            text_feature = text_feature.to(parts_feature[0].device).float()
            motion_feature_global = self.motion_text_proj(global_feat)  # [B, d_model]
            motion_query = self.motion_all_proj(feature_quantized)
            # 跨模态注意力
            if motion_mask is not None:
                attn_output, _ = self.cross_attention(
                    query=text_feature,
                    key=motion_query,
                    value=motion_query,
                    key_padding_mask=~motion_mask
                )  # [bs, seq_text, bert_hidden]
            else:
                attn_output, _ = self.cross_attention(
                    query=text_feature,
                    key=motion_query,
                    value=motion_query
                )
            fused_features = self.temporal_fusion(
                attn_output,
                src_key_padding_mask=~attention_mask
            )  # [bs, seq_text, bert_hidden]
            logits = self.mlm_head(fused_features)  # [bs, seq_text, vocab_size]
            
            # 计算掩码损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            mlm_loss = loss_fct(active_logits, active_labels.long())
            text_feature_pooler = self.text_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return feature_quantized, [contrastive_loss, mlm_loss]

class Dualsem_encoderv2(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=4,  # 减少注意力头数
                 num_layers=1,  # 减少Transformer层数
                 bert_hidden_dim=768,
                 vocab_size=30522,
                 dropout=0.2,   # 增加dropout率
                 down_sample = False):  
        super().__init__()
        # 改进1: 简化结构 + 正则化
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.part_position = nn.Embedding(6, d_model)
        self.vocab_size = vocab_size
        # 改进2: 增加Transformer层的Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,  # 增加dropout
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 改进3: 增强时间建模正则化
        self.time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2*d_model,  # 减少FFN维度
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 改进4: 部分微调BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # 默认冻结


        # 改进5: 加强投影层正则化
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.text_motion_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 改进6: 增强跨模态注意力正则化
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=4,  # 减少注意力头
                dim_feedforward=2*bert_hidden_dim,  # 降低FFN维度
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 减少层数
        ])

        # 改进7: 对比学习增强
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        
        # 改进8: 数据增强
        self.motion_aug = ComposeAugmentation([
            # TemporalCrop(max_ratio=0.2),
            FeatureJitter(std=0.05)
        ])
        
        # 改进9: 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            # nn.LayerNorm(bert_hidden_dim),
            # nn.Dropout(dropout),
            # nn.GELU()
        )
        
        # 改进10: MLM head with label smoothing
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        
        # 改进11: 运动文本投影
        self.motion_text_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.sem_quantizer = QuantizeEMAReset(args.vqvae_sem_nb, d_model, args)
        self.ifdown_sample = down_sample
        if down_sample:
            self.down_sample = TemporalDownsamplerV3(d_model)
        else:
            self.down_sample = nn.Identity()

    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        """
        计算动作和文本之间的Top-K匹配
        Args:
            motion: 动作特征列表 [6, B, T, D]
            text: 文本字符串
            motion_mask: 动作掩码 [B, T]
            topk: 返回的top-k结果数
            text_mask: 文本掩码字典
        Returns:
            [r1, r3, r5]: 召回率指标
            [r1_mlm, r3_mlm, r5_mlm]: MLM任务的召回率指标
        """
        # 部件特征预处理
        B, T = motion[0].shape[0], motion[0].shape[1]
        if self.ifdown_sample:
            T = T // 4
            
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(motion):
            part_embeds.append(self.down_sample(feat + self.part_position.weight[i][None, None, :]))
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        
            
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            time_feat = self.time_transformer(time_feat)
            
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        
        # 使用sem_quantizer进行特征量化
        cls_token, _, _ = self.sem_quantizer(feature[:, :, 0, :].permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)
        
        # 文本特征提取
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_motion_proj(text_feature)
        
        # 计算相似度矩阵
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        # MLM任务的召回率计算
        if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion[0].device)
            labels = text_mask['labels'].to(motion[0].device).float()
            attention_mask = text_mask['attention_mask'].to(motion[0].device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion[0].device).float()
                text_feature_pooler = text_mask['feature'].to(motion[0].device).float()
                
            # 特征投影
            text_query = self.text_proj(text_feature)
            motion_query = self.motion_all_proj(cls_token)
            
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 计算MLM任务的Top-K召回率
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
            active_labels = active_labels.long()  # [active_num]
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            r1_mlm = hits[:, 0].sum().float() / active_labels.size(0)
            r3_mlm = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5_mlm = hits.sum().float() / active_labels.size(0)
            
            return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], \
                   [r1_mlm.cpu().item(), r3_mlm.cpu().item(), r5_mlm.cpu().item()]
                   
        return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], [0, 0, 0]

    def forward(self, parts_feature, text=None, text_mask=None, motion_mask=None):
        # 部件特征预处理 bs,6,seq,d
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        if self.ifdown_sample:
            T = T // 4
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(parts_feature):
            part_embeds.append(self.down_sample(feat + self.part_position.weight[i][None, None, :]))
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        # spatial_cube = self.down_sample(spatial_cube)
        # 数据增强
        if self.training:
            spatial_cube = self.motion_aug(spatial_cube)
            
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            time_feat = self.time_transformer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            time_feat = self.time_transformer(time_feat)
            
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        
        cls_token, loss_commit, perplexity = self.sem_quantizer(feature[:, :, 0, :].permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
                    
            # 特征投影
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(cls_token)
            
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
            mlm_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return cls_token, [contrastive_loss, mlm_loss], [loss_commit, perplexity]

class Dualsem_encoderv3(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=4,  # 减少注意力头数
                 num_layers=2,  # 减少Transformer层数
                 bert_hidden_dim=768,
                 vocab_size=30522,
                 dropout=0.2,   # 增加dropout率
                 down_sample = False,
                 causal = False):  
        super().__init__()
        # 改进1: 简化结构 + 正则化
        self.global_part_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.part_position = nn.Embedding(6, d_model)
        self.vocab_size = vocab_size
        # 改进2: 增加Transformer层的Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,  # 增加dropout
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 添加时间降采样层
        self.ifdown_sample = down_sample
        if down_sample:
            self.time_downsamplers = nn.ModuleList([
                TemporalDownsamplerHalf(d_model) for _ in range(num_layers)
            ])
        else:
            self.time_downsamplers = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])
        
        if causal:
            self.time_transformer = CausalTransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=2*d_model,  # 减少FFN维度
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        else:
            # 改进3: 增强时间建模正则化
            self.time_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=2*d_model,  # 减少FFN维度
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        

        # 改进4: 部分微调BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # 默认冻结


        # 改进5: 加强投影层正则化
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.text_motion_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 改进6: 增强跨模态注意力正则化
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=4,  # 减少注意力头
                dim_feedforward=2*bert_hidden_dim,  # 降低FFN维度
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 减少层数
        ])

        # 改进7: 对比学习增强
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        
        # 改进8: 数据增强
        self.motion_aug = ComposeAugmentation([
            # TemporalCrop(max_ratio=0.2),
            FeatureJitter(std=0.05)
        ])
        
        # 改进9: 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            # nn.LayerNorm(bert_hidden_dim),
            # nn.Dropout(dropout),
            # nn.GELU()
        )
        
        # 改进10: MLM head with label smoothing
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        
        # 改进11: 运动文本投影
        self.motion_text_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.sem_quantizer = QuantizeEMAReset(args.vqvae_sem_nb, d_model, args)

    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        """
        计算动作和文本之间的Top-K匹配
        Args:
            motion: 动作特征列表 [6, B, T, D]
            text: 文本字符串
            motion_mask: 动作掩码 [B, T]
            topk: 返回的top-k结果数
            text_mask: 文本掩码字典
        Returns:
            [r1, r3, r5]: 召回率指标
            [r1_mlm, r3_mlm, r5_mlm]: MLM任务的召回率指标
        """
        # 部件特征预处理
        B, T = motion[0].shape[0], motion[0].shape[1]
        
            
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(motion):
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        
        
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        
        if self.ifdown_sample:
            T = T // 4
        
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                if self.ifdown_sample:
                    time_key_padding_mask = time_key_padding_mask[:, ::2]  # 更新mask
                time_feat = layer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
        
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        
        # 使用sem_quantizer进行特征量化
        cls_token, _, _ = self.sem_quantizer(feature[:, :, 0, :].permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)
        
        # 文本特征提取
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_motion_proj(text_feature)
        
        # 计算相似度矩阵
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        # MLM任务的召回率计算
        if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion[0].device)
            labels = text_mask['labels'].to(motion[0].device).float()
            attention_mask = text_mask['attention_mask'].to(motion[0].device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion[0].device).float()
                text_feature_pooler = text_mask['feature'].to(motion[0].device).float()
                
            # 特征投影
            text_query = self.text_proj(text_feature)
            motion_query = self.motion_all_proj(cls_token)
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 计算MLM任务的Top-K召回率
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
            active_labels = active_labels.long()  # [active_num]
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            r1_mlm = hits[:, 0].sum().float() / active_labels.size(0)
            r3_mlm = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5_mlm = hits.sum().float() / active_labels.size(0)
            
            return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], \
                   [r1_mlm.cpu().item(), r3_mlm.cpu().item(), r5_mlm.cpu().item()]
                   
        return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], [0, 0, 0]

    def forward(self, parts_feature, text=None, text_mask=None, motion_mask=None):
        # 部件特征预处理 bs,6,seq,d
        B, T = parts_feature[0].shape[0], parts_feature[0].shape[1]
        
        # 时空位置编码注入
        part_embeds = []
        for i, feat in enumerate(parts_feature):
            part_embeds.append(feat + self.part_position.weight[i][None, None, :])
            
        # 构建时空特征立方体
        spatial_cube = torch.stack(part_embeds, dim=2)
        # 数据增强
        if self.training:
            spatial_cube = self.motion_aug(spatial_cube)
            
        # 添加全局token
        global_part_tokens = self.global_part_token.expand(B*T, -1, -1)
        fused_feat = torch.cat([
            global_part_tokens,
            rearrange(spatial_cube, 'b t p d -> (b t) p d', b=B)
        ], dim=1)
        
        # 空间特征处理
        spatial_feat = rearrange(fused_feat, '(b t) p d-> (b t) p d', b=B, p=7)
        spatial_feat = self.spatial_transformer(spatial_feat)
        # 时间特征处理
        time_feat = rearrange(spatial_feat, '(b t) p d-> (b p) t d', b=B, p=7)
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                if self.ifdown_sample:
                    time_key_padding_mask = time_key_padding_mask[:, ::2]  # 更新mask
                time_feat = layer(time_feat, src_key_padding_mask=~time_key_padding_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
            
        # 特征重组
        feature = rearrange(time_feat, '(b p) t d -> b t p d', b=B, p=7)
        
        cls_token, loss_commit, perplexity = self.sem_quantizer(feature[:, :, 0, :].permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        
        if text is not None:
            text_feature, text_id = text
            if text_mask is not None:
                input_ids = text_mask['input_ids'].to(parts_feature[0].device)
                labels = text_mask['labels'].to(parts_feature[0].device).float()
                attention_mask = text_mask['attention_mask'].to(parts_feature[0].device).bool()
                
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_feature = bert_outputs.last_hidden_state.to(parts_feature[0].device).float()
                    text_feature_pooler = text_mask['feature'].to(parts_feature[0].device).float()
                    
            # 特征投影
            text_feature = text_feature.to(parts_feature[0].device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(cls_token)
            if self.ifdown_sample:
                motion_mask = motion_mask[:, ::4]
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(parts_feature[0].device)
            mlm_loss = torch.tensor(0.0).to(parts_feature[0].device)
        
        return cls_token, [contrastive_loss, mlm_loss], [loss_commit, perplexity]

class TemporalDownsamplerCausalV3(nn.Module):
    """时间维度1/2降采样模块，使用多尺度卷积和残差连接保留更多特征信息"""
    def __init__(self, d_model, causal=False):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        
        # 特征扩展层
        self.expand = nn.Conv1d(d_model, d_model * 2, 1)
        
        # 多尺度卷积分支
        if causal:
            self.branch1 = nn.Sequential(
                RepeatFirstElementPad1d(padding=2),
                nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, stride=2, padding=0),
                nn.GELU(),
                nn.LayerNorm([d_model * 2, 1])
            )
            self.branch2 = nn.Sequential(
                RepeatFirstElementPad1d(padding=3),
                nn.Conv1d(d_model * 2, d_model * 2, kernel_size=5, stride=2, padding=0),
                nn.GELU(),
                nn.LayerNorm([d_model * 2, 1])
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.LayerNorm([d_model * 2, 1])
            )
            self.branch2 = nn.Sequential(
                nn.Conv1d(d_model * 2, d_model * 2, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.LayerNorm([d_model * 2, 1])
            )
            
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * 4, d_model * 2, 1),
            nn.GELU(),
            nn.LayerNorm([d_model * 2, 1]),
            nn.Conv1d(d_model * 2, d_model, 1)
        )
        
        # 直接下采样分支（残差连接）
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//2, C]
        """
        identity = x
        
        # 转换维度并扩展特征
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.expand(x)
        
        # 多尺度特征提取
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        
        # 特征融合
        y = torch.cat([y1, y2], dim=1)
        y = self.fusion(y)
        
        # 残差连接
        identity = identity.permute(0, 2, 1)
        identity = self.downsample(identity)
        y = y + identity
        
        # 转换回原始维度顺序
        y = y.permute(0, 2, 1)  # [B, T//2, C]
        
        return y

if __name__ == '__main__':
    model = TemporalDownsamplerV3(d_model=256, causal=True)
    x = torch.randn(1, 10, 256)  # [B, T, C]
    out = model(x)
    print(out.shape)
    model = TemporalDownsamplerV3(d_model=256, causal=False)
    x = torch.randn(1, 10, 256)  # [B, T, C]
    out = model(x)
    print(out.shape)