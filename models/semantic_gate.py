import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantize_cnn import QuantizeEMAReset
from models.vqvae_bodypart import VQVAE_bodypart
import options.option_vq_bodypart as option_vq
from options.option_vq_bodypart import vqvae_bodypart_cfg

PART_DESCRIPTIONS = {
    'Root': [
        "body's central movement and balance control",
        "core motion of the entire body",
        "root position and orientation in space",
        "global translation and rotation"
    ],
    'R_Leg': [
        "right leg stepping and kicking motions",
        "right foot contact with ground",
        "right knee bending and stretching",
        "right leg weight bearing movements"
    ],
    'L_Leg': [
        "left leg walking and running patterns",
        "left foot lifting and placing",
        "left hip swinging motions",
        "left leg balance maintenance"
    ],
    'Backbone': [
        "spine bending forward and backward",
        "torso twisting and rotation",
        "upper body posture control",
        "chest and abdomen movements"
    ],
    'R_Arm': [
        "right arm swinging and waving",
        "right hand grasping and releasing",
        "right elbow flexion and extension",
        "right shoulder rotation"
    ],
    'L_Arm': [
        "left arm gesturing and pointing",
        "left hand manipulating objects",
        "left arm elevation and depression",
        "left wrist articulation"
    ]
}

# from transformers import AutoModel, AutoTokenizer
from transformers import CLIPModel, CLIPTokenizer
class PartSemanticEncoder(nn.Module):
    """部位语义编码模块"""
    def __init__(self, part_names, text_dim=512):
        super().__init__()
        self.part_names = part_names
        
        # 加载预训练文本模型
        # self.text_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.text_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        
        # 预计算部位描述特征
        self.desc_embeddings = nn.ParameterDict({
            name: nn.Parameter(self._encode_text(desc_list)) 
            for name, desc_list in PART_DESCRIPTIONS.items()
        })
        
        # 文本特征适配器
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 256)
        )
        
        # del self.text_model, self.tokenizer  # 释放内存
        # for param in self.text_model.parameters():
            # param.requires_grad = False
        # for param in self.tokenizer.parameters():
        #     param.requires_grad = False
    
    def _encode_text(self, texts):
        """编码描述文本"""
        # inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77).to(next(self.parameters()).device)
        with torch.no_grad():
            # outputs = self.text_model(**inputs).last_hidden_state.mean(dim=1)
            outputs = self.text_model.get_text_features(**inputs)
        return outputs #.mean(dim=0)  # 平均所有描述
    
    def forward(self, text_tokens):
        """
        input_text: List[str] 输入文本列表
        返回: 
            text_emb: [B,256] 投影后的文本特征
            part_sims: [B,6] 文本-部位相似度
        """
        # # 编码输入文本
        # batch_max = 64
        # text_feat_all = []
        # num_s = (len(input_text) + batch_max - 1) // batch_max 
        # for i in range(num_s):
        #     start_id = i * batch_max
        #     end_id = min((i+1)*batch_max, len(input_text))
        #     input_text_tmp = input_text[start_id:end_id]
        #     inputs = self.tokenizer(input_text_tmp, return_tensors='pt', padding=True, truncation=True, max_length=77).to(next(self.parameters()).device)
        #     text_feat_all.append(self.text_model.get_text_features(**inputs))
        # text_feat = torch.cat(text_feat_all, dim=0)
        # inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=77).to(next(self.parameters()).device)
        # text_feat = self.text_model.get_text_features(**inputs)
        if len(text_tokens.shape)== 3:
            text_tokens = text_tokens.squeeze(1)
        text_feat = text_tokens
        # 投影到公共空间
        proj_text = self.text_proj(text_feat)  # [B,256]
        
        # 计算与各部位描述的相似度
        sim_list = {}
        # score = []
        part_scores_list = []
        for i in range(4):
            for name in self.part_names:
                desc_feat = self.text_proj(self.desc_embeddings[name])  # [256]
                score = F.cosine_similarity(proj_text, desc_feat[i].unsqueeze(0), dim=-1)
            # sim_list.append(sim)
                sim_list[name] = score
            # sim_list[name] = torch.cat(score, dim=0)
        
            # part_sims = torch.stack(sim_list, dim=-1)  # [B,6]
            part_sims = torch.stack([sim_list[name] for name in self.part_names], dim=-1)
            part_scores_list.append(F.softmax(part_sims, dim=-1))
        part_scores = torch.stack(part_scores_list, dim=0).max(dim=0).values
        # part_name_scores = {name: score for name, score in zip(self.part_names, part_scores.squeeze().tolist())}
        return proj_text, part_scores

class SemanticAwareQuantizer(nn.Module):
    """集成EMA码本的语义感知量化器"""
    def __init__(self, ori_codebook: QuantizeEMAReset, code_dim, part_name):
        super().__init__()
        self.ori_codebook = ori_codebook
        self.code_dim = code_dim
        self.part_name = part_name
        
        # 语义门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(256 + code_dim, 512),  # 文本特征 + 码本特征
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, code_dim),
            nn.Sigmoid()
        )
        
        # 初始化原始码本为不可训练
        for param in self.ori_codebook.parameters():
            param.requires_grad = False

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
        self.semantic_encoder = PartSemanticEncoder(self.parts_name)
        self.quantizers = nn.ModuleDict({
            name: SemanticAwareQuantizer(
                # num_codes=parts_code_nb[name],
                code_dim=parts_code_dim[name],
                part_name=name,
                ori_codebook=getattr(self, f'quantizer_{name}')
            ) for name in self.parts_name
        })
        
        for name in self.parts_name:
            for param in getattr(self, f'enc_{name}').parameters():
                param.requires_grad = False
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
    
    def get_semantic_scores(self, text_tokens):
        """获取文本-部位相似度分数"""
        with torch.no_grad():
            _, part_scores = self.semantic_encoder(text_tokens)
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
        if text_tokens is not None:
            text_feat, part_sims = self.semantic_encoder(text_tokens)
        
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder = encoder(x_in)

            # Quantization
            # quantizer = getattr(self, f'quantizer_{name}')
            if text_tokens is not None:
                x_quantized, loss, perplexity = self.quantizers[name](x_encoder, text_feat, part_sims[:,self.parts_name.index(name)])
            else:
                x_quantized, loss, perplexity = self.quantizers[name](x_encoder)
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
    # def forward(self, motion, text=None):
    #     # 编码运动特征
    #     part_features = self.encoder(motion)  # {name: [B,T,D]}
        
    #     # 文本语义处理
    #     if text is not None:
    #         text_feat, part_sims = self.semantic_encoder(text)
            
    #         # 分部位量化
    #         quantized = {}
    #         for idx, name in enumerate(self.part_names):
    #             sim_weight = part_sims[:, idx].unsqueeze(1)  # [B,1]
    #             quantized[name] = self.quantizers[name](
    #                 part_features[name],
    #                 text_feat,
    #                 sim_weight
    #             )
    #     else:
    #         # 无文本时原始量化
    #         quantized = {name: self.quantizers[name](part_features[name], None, None) 
    #                     for name in self.part_names}
        
    #     # 解码重建
    #     recon_motion = self.decoder(quantized)
    #     return recon_motion
    
    def compute_loss(self, motion, recon, text=None):
        # 重建损失
        recon_loss = F.l1_loss(recon, motion)
        
        # 对比损失
        contrast_loss = 0
        if text is not None:
            text_feat, part_sims = self.semantic_encoder(text)
            batch_size = text_feat.size(0)
            
            # 构造正负样本对
            pos_sim = torch.diag(F.cosine_similarity(
                text_feat, 
                torch.stack([self.semantic_encoder.desc_embeddings[name] for name in self.part_names], dim=0)[torch.argmax(part_sims, dim=-1)]
            ))
            
            neg_sim = F.cosine_similarity(
                text_feat.unsqueeze(1),
                torch.stack(list(self.semantic_encoder.desc_embeddings.values())).unsqueeze(0),
                dim=-1
            )
            neg_sim = neg_sim.mean(dim=-1)
            
            contrast_loss = F.margin_ranking_loss(
                pos_sim,
                neg_sim,
                target=torch.ones_like(pos_sim),
                margin=0.3
            )
        
        return 0.7*recon_loss + 0.3*contrast_loss
    
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
    net.load_checkpoint('output/ParCo_official_HumanML3D/VQVAE-ParCo-t2m-default/net_last.pth')
    args.resume_pth = "output/00048-t2m-ParCo/VQVAE-ParCo-t2m-default/net_best_div.pth"
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
    