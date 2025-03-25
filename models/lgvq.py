import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from transformers import CLIPModel, CLIPTokenizer
import clip
from models.quantize_cnn import QuantizeEMAReset
# from models.rvqvae_bodypart import CausalTransformerEncoder, ContrastiveLossWithSTSV2
from models.vqvae_bodypart import VQVAE_bodypart

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

    # def _generate_cross_mask(self, text_mask: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
    #     """生成联合时空注意力掩码"""
    #     if text_mask is None or motion_mask is None:
    #         return None
        
    #     # 文本维度扩展 [B, seq_text] -> [B, seq_text, 1]
    #     text_pad = ~text_mask.unsqueeze(-1)
    #     # 运动维度扩展 [B, seq_motion] -> [B, 1, seq_motion]
    #     motion_pad = ~motion_mask.unsqueeze(1)
        
    #     # 联合掩码：任一位置为pad则屏蔽 [B, seq_text, seq_motion]
    #     combined_mask = text_pad | motion_pad
        
    #     # 适配多头注意力 [B*num_heads, seq_text, seq_motion]
    #     return combined_mask.repeat_interleave(8, dim=0)

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
        feature = rearrange(time_feat, '(b p) t d -> p b t d', b=B, p=7)
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