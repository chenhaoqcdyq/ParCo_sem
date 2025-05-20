import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantize_cnn import QuantizeEMAReset
from models.vqvae_bodypart import VQVAE_bodypart
import options.option_vq_bodypart as option_vq
from options.option_vq_bodypart import vqvae_bodypart_cfg
from transformers import CLIPModel, CLIPTokenizer

# class PositionalEncoding(nn.Module):
#     """Transformer的位置编码模块"""
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(1), :]
#         return self.dropout(x)

class PartSemanticEncoder(nn.Module):
    """基于CLIP的语义编码器（适配预训练文本特征）"""
    def __init__(self, parts_output_dim, parts_name, clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.parts_name = parts_name
        self.num_parts = len(parts_name)
        self.parts_output_dim = parts_output_dim
        
        # 加载预训练CLIP模型
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # 冻结CLIP参数（根据需求可选）
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # # 可学习的部位嵌入（与CLIP文本特征维度对齐）
        # self.part_embeddings = nn.Parameter(
        #     torch.randn(self.num_parts, self.clip.config.text_config.hidden_size)
        # )
        self.part_proj = nn.ModuleDict(
            {name: nn.Linear(self.parts_output_dim[name], 512) for name in self.parts_name}
        )
        
        # 文本特征增强投影
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip.config.text_config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.tau = nn.Parameter(torch.ones([]) * 0.07)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def encode_text(self, text):
        """使用CLIP编码文本"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}
        text_features = self.clip.get_text_features(**inputs)
        return text_features

    def forward(self, motion_encoded, text_input):
        """
        前向传播（支持直接输入文本或预提取特征）
        :param text_input: 可以是文本列表或预提取的CLIP特征（B, 512）
        :return: 
            text_feat: 增强后的文本特征 (B, 512)
            part_sims: 部位相似度分数 (B, num_parts)
        """
        # 自动判断输入类型
        if isinstance(text_input, list) or isinstance(text_input, str):
            # 输入为原始文本，使用CLIP编码
            text_feat = self.encode_text(text_input)
        else:
            # 输入为预提取特征
            if len(text_input.shape) == 3 and text_input.shape[1] ==1:
                text_feat = text_input.squeeze(1)
            else:
                text_feat = text_input
            
        # 特征增强
        enhanced_feat = self.text_proj(text_feat)  # (B, 512)
        motion_feats = {name: self.pool(self.part_proj[name](motion_encoded[name].permute(0, 2, 1)).permute(0, 2, 1)).squeeze(-1) for name in self.parts_name}
        # 计算跨模态相似度
        sim_matrix = torch.stack([
            F.cosine_similarity(text_feat.unsqueeze(1), feat.unsqueeze(0), dim=-1)
            for feat in motion_feats.values()
        ], dim=-1)  # (B, B, num_parts)
        
        # 对角线元素为各部位匹配分数
        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        part_sims = sim_matrix[diag_mask].view(-1, self.num_parts)  # (B, num_parts)
        
        # 温度缩放
        part_sims = part_sims / self.tau.clamp(min=0.01)
        part_sims = F.softmax(part_sims, dim=-1)
        
        # part_sims = F.softmax(sim_scores, dim=-1)
        
        return enhanced_feat, part_sims

class SemanticAwareQuantizer(nn.Module):
    """集成EMA码本的语义感知量化器"""
    def __init__(self, ori_codebook: QuantizeEMAReset, code_dim, part_name):
        super().__init__()
        self.ori_codebook = ori_codebook
        self.code_dim = code_dim
        self.part_name = part_name
        
        # 语义门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(512 + code_dim, 512),  # 文本特征 + 码本特征
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, code_dim),
            nn.Sigmoid()
        )
        
        # 初始化原始码本为不可训练
        # for param in self.ori_codebook.parameters():
        #     param.requires_grad = False

    def load_checkpoint(self, checkpoint):
        """加载预训练码本"""
        param = torch.load(checkpoint)
        new_param = {}
        for key, value in param['net'].items():
            if f'vqvae.quantizer_{self.part_name}' in key:
                new_key = key.replace(f'vqvae.quantizer_{self.part_name}.', '')
                new_param[new_key] = value
        self.ori_codebook.load_state_dict(new_param)


    def _modify_codebook(self, text_feat, part_sim):
        """生成临时语义增强码本"""
        orig_code = self.ori_codebook.codebook.detach()  # [N,D]
        
        # 生成门控系数
        expand_text = text_feat.unsqueeze(1)  # [B,1,256]
        gate_input = torch.cat([
            expand_text.expand(-1, orig_code.size(0), -1),
            orig_code.unsqueeze(0).expand(text_feat.size(0), -1, -1)
        ], dim=-1)
        gate = self.gate_net(gate_input)  # [B,N,D]
        
        # 加权更新
        update_weight = part_sim.unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        modified_code = orig_code + (update_weight * gate).mean(dim=0)
        
        return 0.7*orig_code + 0.3*modified_code  # 保持原始码本主导

    def forward(self, x, text_feat=None, part_sim=None):
        """
        x: [N, C, T] 输入特征
        text_feat: [B,256] 文本特征 (当使用语义门控时)
        part_sim: [B,1] 部位相似度权重
        """
        # 原始量化流程
        x_d, commit_loss, perplexity = self.ori_codebook(x)
        
        # 语义增强分支
        if text_feat is not None and part_sim is not None:
            B = x.size(0)
            
            # 生成临时码本
            temp_codebook = self._modify_codebook(text_feat, part_sim)
            
            # 重新量化
            x_flat = self.ori_codebook.preprocess(x)  # [NT, C]
            
            # 使用临时码本计算距离
            distance = torch.cdist(x_flat, temp_codebook)
            _, code_idx = torch.min(distance, dim=-1)
            
            # 解码时混合原始码本和临时码本
            alpha = 0.3  # 语义增强权重
            x_d_mix = (1-alpha)*F.embedding(code_idx, self.ori_codebook.codebook) + \
                      alpha*F.embedding(code_idx, temp_codebook)
            
            # 保持梯度流
            x_d = x_flat + (x_d_mix - x_flat).detach()
            
            # 恢复形状
            x_d = x_d.view(x.size(0), x.size(2), -1).permute(0, 2, 1).contiguous()
            
            # 计算语义commit损失
            sem_commit_loss = F.mse_loss(x_flat, x_d_mix.detach())
            commit_loss = 0.7*commit_loss + 0.3*sem_commit_loss

        return x_d, commit_loss, perplexity

    def get_codebook(self):
        """获取当前码本（包含EMA更新）"""
        return self.ori_codebook.codebook
    
    def dequantize(self, x):
        """解码量化特征"""
        return self.ori_codebook.dequantize(x)
    
    def quantize(self, x):
        """量化特征"""
        return self.ori_codebook.quantize(x)

class SemanticVQVAE(VQVAE_bodypart):
    """完整的语义VQVAE模型"""
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
                 norm=None):
        super().__init__(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, 
                         down_t, stride_t, depth, dilation_growth_rate, activation, norm)
        
        # 语义模块
        self.semantic_encoder = PartSemanticEncoder(self.parts_output_dim, self.parts_name)
        self.quantizers = nn.ModuleDict({
            name: SemanticAwareQuantizer(
                # num_codes=parts_code_nb[name],
                code_dim=parts_code_dim[name],
                part_name=name,
                ori_codebook=getattr(self, f'quantizer_{name}')
            ) for name in self.parts_name
        })
        
        # for name in self.parts_name:
        #     for param in getattr(self, f'enc_{name}').parameters():
        #         param.requires_grad = False
            # for param in getattr(self, f'dec_{name}').parameters():
            #     param.requires_grad = False
    
    def load_checkpoint(self, checkpoint):
        """加载预训练模型"""
        param = torch.load(checkpoint)
        new_param = {}
        for key, value in param['net'].items():
            if 'vqvae.' in key:
                new_key = key.replace('vqvae.', '')
                new_param[new_key] = value
        self.load_state_dict(new_param, strict=False)
    
    def get_semantic_scores(self, text_input):
        """直接支持文本或特征输入"""
        with torch.no_grad():
            _, part_scores = self.semantic_encoder(text_input)
        return part_scores
    
    def forward(self, parts, text_tokens=None):
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
        # 文本语义处理
        
        x_encoder = {}
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder[name] = encoder(x_in)

        if text_tokens is not None:
            text_feat, part_sims = self.semantic_encoder(x_encoder, text_tokens)

        for i, name in enumerate(self.parts_name):
            # Quantization
            if text_tokens is not None:
                x_quantized, loss, perplexity = self.quantizers[name](x_encoder[name], text_feat, part_sims[:,self.parts_name.index(name)])
            else:
                x_quantized, loss, perplexity = self.quantizers[name](x_encoder[name])
            # x_quantized, loss, perplexity = self.quantizers[name](x_encoder, text_feat, part_sims)

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
    
    
    def forward_decoder(self, parts, text=None):
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
        # 文本语义处理
        if text is not None:
            text_feat, part_sims = self.semantic_encoder(text)
            
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            assert x.shape[0] == 1  # ensure batch size is 1
            codes_len = x.shape[1]
            assert codes_len == base_codes_len  # make sure all parts has same codes_len

            # quantizer = getattr(self, f'quantizer_{name}')
            x_d = self.quantizers[name].dequantize(x)  # (B, codes_len) => (B, codes_len, code_dim), B == 1

            # It seems the .view() operation does not bring any change.
            #   The code probably is just adapted from the quantizer's code
            x_d = x_d.view(1, codes_len, -1).permute(0, 2, 1).contiguous()  # (B, code_dim, codes_len)

            # decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_d)  # (B, raw_motion_dim, seq_len)
            x_out = self.postprocess(x_decoder)  # (B, seq_len, raw_motion_dim)

            parts_out.append(x_out)

        return parts_out

    
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_part_scores(part_scores, text_input, part_names):
    """
    可视化文本-部位关联分数
    part_scores: [B,6] 的相似度矩阵
    text_input: 对应的输入文本列表
    part_names: 部位名称列表
    """
    plt.figure(figsize=(12, 6))
    
    # 转换为numpy并取第一个样本（假设batch_size=1）
    scores = part_scores[0].detach().cpu().numpy()
    
    # 创建柱状图
    ax = sns.barplot(x=part_names, y=scores, palette="viridis")
    
    # 设置标题和标签
    ax.set_title(f"Text-Part Correlation: '{text_input[0]}'")
    ax.set_ylabel("Semantic Similarity Score")
    ax.set_xlabel("Body Parts")
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig('test.png')


if __name__ == '__main__':
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    args.vqvae_arch_cfg = vqvae_bodypart_cfg[args.vqvae_cfg]
    if args.dataname == 'kit':
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
        
    else :
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22
    net = SemanticVQVAE(
        args,  # use args to define different parameters in different quantizers
        args.vqvae_arch_cfg['parts_code_nb'],
        args.vqvae_arch_cfg['parts_code_dim'],
        args.vqvae_arch_cfg['parts_output_dim'],
        args.vqvae_arch_cfg['parts_hidden_dim'],
        args.down_t,
        args.stride_t,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    )
    # net.load_checkpoint('output/ParCo_official_HumanML3D/VQVAE-ParCo-t2m-default/net_last.pth')
    args.resume_pth = "output/00055-t2m-ParCo/VQVAE-ParCo-t2m-default/net_best_matching.pth"
    if args.resume_pth:
        # logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.cuda()
    net.eval()
    input_text = ["right leg"]
    
    # 编码文本
    text_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_tokens = text_model.get_text_features(**inputs)
    
    # 获取分数
    part_scores = net.get_semantic_scores(text_tokens.cuda())
    
    # 可视化
    part_names = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
    visualize_part_scores(part_scores, input_text, part_names)
    # if args.resume_pth:
    #     # logger.info('loading checkpoint from {}'.format(args.resume_pth))
    #     ckpt = torch.load(args.resume_pth, map_location='cpu')
    #     net.load_state_dict(ckpt['net'], strict=True)
    