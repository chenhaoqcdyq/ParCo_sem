import os 
from os.path import join as pjoin
import json
import re
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

import clip
from tqdm import tqdm

from options.get_eval_option import get_opt
import options.option_transformer_bodypart as option_trans
from options.option_transformer_bodypart import trans_bodypart_cfg

import utils.utils_model as utils_model
import utils.eval_bodypart as eval_bodypart
from utils.word_vectorizer import WordVectorizer
from utils.misc import EasyDict

import models.vqvae_bodypart as vqvae_bodypart
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.t2m_trans_bodypart as trans_bodypart
import models.t2m_trans_multipart as trans_multipart

# from dataset import dataset_TM_train_bodypart
from dataset import dataset_TM_eval_bodypart
from dataset import dataset_tokenize_bodypart

import models.rvqvae_bodypart as vqvae



# import warnings
# warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# Load VQVAE and its configs
select_vqvae_ckpt = args.select_vqvae_ckpt
assert select_vqvae_ckpt in [
    'last',  # last  saved ckpt
    'fid',  # best FID ckpt
    'div',  # best diversity ckpt
    'top1',  # best top-1 R-precision
    'matching',  # MM-Dist: Multimodal Distance
]
vqvae_train_dir = args.vqvae_train_dir
# Checkpoint path
if vqvae_train_dir.endswith('.pth'):
    args.vqvae_ckpt_path = vqvae_train_dir
    vqvae_train_dir = os.path.dirname(vqvae_train_dir)
else:
    if select_vqvae_ckpt == 'last':
        args.vqvae_ckpt_path = os.path.join(vqvae_train_dir, 'net_' + select_vqvae_ckpt + '.pth')
    else:
        args.vqvae_ckpt_path = os.path.join(vqvae_train_dir, 'net_best_' + select_vqvae_ckpt + '.pth')

# vqvae training config
print('\nLoading training argument...\n')
args.vqvae_training_options_path = os.path.join(vqvae_train_dir, 'train_config.json')
with open(args.vqvae_training_options_path, 'r') as f:
    vqvae_train_args_dict = json.load(f)  # dict
vqvae_train_args = EasyDict(vqvae_train_args_dict)  # convert dict to easydict for convenience
args.vqvae_train_args = vqvae_train_args


# Get running directory: args.run_dir
prev_run_dirs = []
parent_dir = os.path.dirname(vqvae_train_dir)
if os.path.isdir(parent_dir):
    prev_run_dirs = [x for x in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1


# Preprocess configuration
args.trans_arch_cfg = trans_bodypart_cfg[args.trans_cfg]
if args.no_fuse:
    args.use_fuse = False
else:
    args.use_fuse = True

# todo: move to the config file
if args.use_pkeep_scheduler:
    args.pkeep_scheduler = [0, 100000, 200000]
    args.pkeep_list = [0.5, 0.4, 0.3]
else:
    args.pkeep_scheduler = None

args.masktoken = True

# todo: rename the directory
# todo: prepare the description
desc = args.trans_cfg

args.run_dir = os.path.join(parent_dir, f'{cur_run_id:05d}-Trans-{args.exp_name}-{desc}')

assert not os.path.exists(args.run_dir)
print('Creating output directory...')
os.makedirs(args.run_dir)

# quantized dataset directory
if not args.use_existing_vq_data:
    args.vq_dir = os.path.join(args.run_dir, f'quantized_dataset_{args.dataname}')
    assert not os.path.exists(args.vq_dir)
    print('Creating quantized dataset directory...')
    os.makedirs(args.vq_dir)
else:
    args.vq_dir = args.existing_vq_data_dir
    print('\n\n')
    print('####################################################')
    print('==> Warning: using existing tokenized motion data!  ')
    print('==> It is used for debug, do not use it in training!')
    print('####################################################')
    print('\n\n')


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.run_dir)
writer = SummaryWriter(args.run_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# save the training config
args.args_save_dir = os.path.join(args.run_dir, 'train_config.json')
args_dict = vars(args)
with open(args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)


# [Bodypart VQVAE]: construction, weight loading
print('\n\n===> Constructing VQVAE for quantize...')
net = getattr(vqvae, f'HumanVQVAETransformerV{vqvae_train_args.vision}')(vqvae_train_args,  # use args to define different parameters in different quantizers
        parts_code_nb=vqvae_train_args.vqvae_arch_cfg['parts_code_nb'],
        parts_code_dim=vqvae_train_args.vqvae_arch_cfg['parts_code_dim'],
        parts_output_dim=vqvae_train_args.vqvae_arch_cfg['parts_output_dim'],
        parts_hidden_dim=vqvae_train_args.vqvae_arch_cfg['parts_hidden_dim'],
        down_t=vqvae_train_args.down_t,
        stride_t=vqvae_train_args.stride_t,
        depth=vqvae_train_args.depth,
        dilation_growth_rate=vqvae_train_args.dilation_growth_rate,
        activation=vqvae_train_args.vq_act,
        norm=vqvae_train_args.vq_norm
    )    
print('===> Loading VQVAE checkpoint from {}...'.format(args.vqvae_ckpt_path))
ckpt = torch.load(args.vqvae_ckpt_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()


##### ---- Prepare tokenized motion dataset ---- #####
print('\n\n===> Preparing the quantized motion for training:')
if vqvae_train_args.down_vqvae == 1:
    unit_length = 2**vqvae_train_args.down_t
else:
    unit_length = 1
if not args.use_existing_vq_data:
    # [token loader]
    print('===> Constructing dataset for tokenize...')
    train_loader_token = dataset_tokenize_bodypart.DATALoader(
        args.dataname, 1, unit_length=unit_length)

    # Get code (quantized motion)
    print('===> Getting the code...')
    for batch in tqdm(train_loader_token, desc='Tokenizing...'):
        '''
        Batch_size == 1
        Root:     (B, nframes, 7)     >==[quantized]==>  (B, nframes)
        R_Leg:    (B, nframes, 50)    >==[quantized]==>  (B, nframes)
        L_Leg:    (B, nframes, 50)    >==[quantized]==>  (B, nframes)
        Backbone: (B, nframes, 60)    >==[quantized]==>  (B, nframes)
        R_Arm:    (B, nframes, 60)    >==[quantized]==>  (B, nframes)
        L_Arm:    (B, nframes, 60)    >==[quantized]==>  (B, nframes)
        '''

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, name = batch

        bs, seq = Root.shape[0], Root.shape[1]
        parts = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
        for i in range(len(parts)):
            parts[i] = parts[i].cuda().float()

        tokenized_parts = net.encode(parts)

        for i, part_name in enumerate(['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):
            tok_p = tokenized_parts[0][i]  # (B, nframes)
            tok_p = tok_p.cpu().numpy()
            # use name[0] because the batch size is 1
            np.save(pjoin(args.vq_dir, name[0] + '_' + part_name + '.npy'), tok_p)
        if vqvae_train_args.lgvq > 0:
            sem_part = tokenized_parts[1]
            sem_part = sem_part[0].cpu().numpy()
            np.save(pjoin(args.vq_dir, name[0] + '_sem.npy'), sem_part)

else:
    print('\n\n')
    print('####################################################')
    print('==> Warning: using existing tokenized motion data!  ')
    print('==> It is used for debug, do not use it in training!')
    print('####################################################')
    print('\n\n')


##### ---- Dataloader ---- #####

# val loader
print('\n\n===> Constructing dataset for evaluating transformer...')
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval_bodypart.DATALoader(
    args.dataname, False, 32, w_vectorizer)

print('\n\n===> Constructing dataset for training transformer...')
if vqvae_train_args.lgvq>0:
    import dataset.dataset_TM_train_bodypart_sem as dataset_TM_train_bodypart
else:
    import dataset.dataset_TM_train_bodypart as dataset_TM_train_bodypart
# train loader
train_loader = dataset_TM_train_bodypart.DATALoader(
    dataset_name=args.dataname, vq_dir=args.vq_dir, unit_length=unit_length, codebook_size=args.nb_code,
    batch_size=args.batch_size,)
train_loader_iter = dataset_TM_train_bodypart.cycle(train_loader)


# Prepare evaluation wrapper
print('\n\n===> Loading EvaluatorModelWrapper...')
dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD01/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Network ---- #####
print('\n\n===> Loading CLIP model...')
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False


# [Transformer]: construction, weight loading (if resume)
print('\n\n===> Constructing our bodypart transformer...')
# todo: implement our transformer
# trans_encoder = trans_bodypart.TransformerFuseHiddenDim(

#     clip_dim=args.clip_dim,
#     block_size=args.block_size,
#     num_layers=args.num_layers,
#     n_head=args.n_head_gpt,
#     drop_out_rate=args.drop_out_rate,
#     fc_rate=args.ff_rate,

#     # FusionModule
#     use_fuse=args.use_fuse,
#     fuse_ver=args.fuse_ver,
#     alpha=args.alpha,

#     parts_code_nb=args.trans_arch_cfg['parts_code_nb'],
#     parts_embed_dim=args.trans_arch_cfg['parts_embed_dim'],
#     num_mlp_layers=args.trans_arch_cfg['num_mlp_layers'],
#     fusev2_sub_mlp_out_features=args.trans_arch_cfg['fusev2_sub_mlp_out_features'],
#     fusev2_sub_mlp_num_layers=args.trans_arch_cfg['fusev2_sub_mlp_num_layers'],
#     fusev2_head_mlp_num_layers=args.trans_arch_cfg['fusev2_head_mlp_num_layers'],

# )
semantic_len_max = ((196 // unit_length) + 3) // 4 + 1
trans_encoder = trans_multipart.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,
    dual_head_flag=(vqvae_train_args.lgvq>0),
    semantic_len=semantic_len_max,
    num_parts=6,
)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)

trans_encoder.train()
trans_encoder.cuda()


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()
loss_ce_batch = torch.nn.CrossEntropyLoss(ignore_index=train_loader.dataset.mot_pad_idx, reduction='none')

nb_iter = 0
nb_sample_train = 0
nb_sample_sem_train = 0
avg_loss_cls = {
    'Root': 0.,
    'R_Leg': 0.,
    'L_Leg': 0.,
    'Backbone': 0.,
    'R_Arm': 0.,
    'L_Arm': 0.,
    'sem': 0.,
}
avg_acc_determine = {
    'Root': 0.,
    'R_Leg': 0.,
    'L_Leg': 0.,
    'Backbone': 0.,
    'R_Arm': 0.,
    'L_Arm': 0.,
    'sem': 0.,
}
avg_acc_sample = {
    'Root': 0.,
    'R_Leg': 0.,
    'L_Leg': 0.,
    'Backbone': 0.,
    'R_Arm': 0.,
    'L_Arm': 0.,
    'sem': 0.,
}
right_num_determine = {
    'Root': 0,
    'R_Leg': 0,
    'L_Leg': 0,
    'Backbone': 0,
    'R_Arm': 0,
    'L_Arm': 0,
    'sem': 0,
}
right_num_sample = {
    'Root': 0,
    'R_Leg': 0,
    'L_Leg': 0,
    'Backbone': 0,
    'R_Arm': 0,
    'L_Arm': 0,
    'sem': 0,
}

        
##### ---- Training ---- #####
print('\n\n===> Pre-evaluation before training...')
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
    eval_bodypart.evaluation_transformer_batch(
        args.run_dir, val_loader, net, trans_encoder, logger, writer,
        0, best_fid=1000, best_iter=0, best_div=100,
        best_top1=0, best_top2=0, best_top3=0, best_matching=100,
        clip_model=clip_model, eval_wrapper=eval_wrapper, semantic_flag=(vqvae_train_args.lgvq>0), draw=False)
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100
time_consuming = {
    'dataloading_time': 0.,
    'text_to_clip_feat_time': 0.,
    'token_aug_time': 0.,
    'forward_time': 0.,
    'loss_compute_time': 0.,
    'backward_time': 0.,
}

print('\n\n===> Start training our bodypart transformer...')
while nb_iter <= args.total_iter:

    if args.use_pkeep_scheduler:
        for i, step in enumerate(args.pkeep_scheduler):
            if step == nb_iter:
                args.pkeep = args.pkeep_list[i]


    start_time = time.time()

    batch = next(train_loader_iter)

    '''
    caption:    Dict, contains strings of text describing motion. Pure text of caption. type: tuple. len: B. Each elem is a string.
    Root:       Tensor: (64, 51)   The tokenized motion sequence. (B, max_token_len + 1), the 1 is End flag.
    R_Leg:      Tensor: (64, 51)
    L_Leg:      Tensor: (64, 51)
    Backbone:   Tensor: (64, 51)
    R_Arm:      Tensor: (64, 51)
    L_Arm:      Tensor: (64, 51)
    token_len:  Tensor: (64)       Describing the length of tokenized motion, not include the End token, has only 1 dim.
    '''
    if len(batch) == 10:
        caption, Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, token_len, sem_token, sem_token_len = batch
        sem_token = sem_token.cuda()
        sem_token_len = sem_token_len.cuda()
    else:
        caption, Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, token_len = batch
        sem_token, sem_token_len = None, None

    gt_parts = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
    for i in range(len(gt_parts)):
        gt_parts[i] = gt_parts[i].cuda()
    m_tokens_len = token_len.cuda()
    bs = m_tokens_len.shape[0]

    '''data loading time'''
    dataloading_time = time.time() - start_time
    start_time = time.time()

    if args.classfg > 0:
        # 生成mask来决定哪些样本的文本被替换为空
        mask = torch.bernoulli(args.classfg * torch.ones(len(caption), device='cuda'))
        # 创建新的文本列表，将被mask的文本替换为空字符串
        caption = ["" if mask[i] else caption[i] for i in range(len(caption))]
    text = clip.tokenize(caption, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()  # encode text tokens. (B, 512). 512: clip out feat dim.

    '''text-to-clip feature time'''
    text_to_clip_feat_time = time.time() - start_time
    start_time = time.time()

    input_parts = []

    # [Mask the sequence using a specific mask token]
    mask_token_value = train_loader.dataset.mot_pad_idx + 1  # should be 514(511+3) if the num_vq is 512

    if args.sync_part_maskaug:

        input_p = gt_parts[0][:, :-1]  # used for getting the shape and device
        if args.pkeep == -1:  # pkeep: random mask rate
            proba = np.random.rand(1)[0]
            # The input of torch.bernoulli is the probability of returning 1.
            #  Returning 0 according to probability of (1 - prob)
            mask = torch.bernoulli(proba * torch.ones(input_p.shape, device=input_p.device))
        else:
            mask = torch.bernoulli(args.pkeep * torch.ones(input_p.shape, device=input_p.device))
        mask = mask.round().to(dtype=torch.int64)

        for p in gt_parts:
            # Drop the last token (end or padding flag) in input motion tokens.
            # It's okay to drop the End token in input because it reaches the max length,
            #   and the output need to predict the End token

            input_p = p[:, :-1]  # (B, 50)
            # r_indices = torch.randint_like(input_p, args.nb_code)  # random_indices
            a_indices = mask*input_p + (1-mask)*mask_token_value  # augmented indices

            input_parts.append(a_indices)  # collect augmented input indices

    else:  # default
        for p in gt_parts:
            # Drop the last token (end or padding flag) in input motion tokens.
            # It's okay to drop the End token in input because it reaches the max length,
            #   and the output need to predict the End token

            input_p = p[:, :-1]  # (B, 50)

            # [Data Augmentation for the input tokenized motion sequence ]
            # The following is "Corrupted sequences for the training-testing discrepancy" in T2M-GPT paper.
            #  It randomly replace some tokens with random token (like adding some noises as data augmentation)
            if args.pkeep == -1:  # pkeep: random mask rate
                proba = np.random.rand(1)[0]
                # The input of torch.bernoulli is the probability of returning 1.
                #  Returning 0 according to probability of (1 - prob)
                mask = torch.bernoulli(proba * torch.ones(input_p.shape, device=input_p.device))
            else:
                mask = torch.bernoulli(args.pkeep * torch.ones(input_p.shape, device=input_p.device))

            mask = mask.round().to(dtype=torch.int64)
            if args.rand_index == 1:
                r_indices = torch.randint_like(input_p, args.nb_code)  # random_indices
                a_indices = mask*input_p + (1-mask)*r_indices  # augmented indices
            else:
                a_indices = mask*input_p + (1-mask)*mask_token_value  # augmented indices

            input_parts.append(a_indices)  # collect augmented input indices
        if sem_token is not None:
            input_sem = sem_token
            if args.pkeep == -1:
                proba = np.random.rand(1)[0]
                mask = torch.bernoulli(proba * torch.ones(input_sem.shape, device=input_sem.device))
            else:
                mask = torch.bernoulli(args.pkeep * torch.ones(input_sem.shape, device=input_sem.device))
            mask = mask.round().to(dtype=torch.int64)
            input_sem = mask*input_sem + (1-mask)*mask_token_value
            
            for i in range(len(input_parts)):
                input_parts[i] = torch.cat([input_sem, input_parts[i]], dim=1)


    '''token augmentation time'''
    token_aug_time = time.time() - start_time
    start_time = time.time()


    '''
    Input:
        input_parts: List, [(B, 50), ..., (B, 50)]          
            50: input motion token length.  
            value range: 0~514 (512 vq_num (0~511) + 1 end flag (512) + 1 padding flag (513) + 1 masktoken flag (514)
        feat_clip_text: (B, 512)    
        
    Output:
        
        parts_cls_pred: List, [(B, 51, 513), ..., (B, 51, 513)]      logits matrix
            513: number of token categories, 512 motion tokens (0~511) + 1 end flag (512)
            token value range: 0~512,
                512: end flag.
                
    The 513-th token: is padding flag
        It is used in the input, but not used in loss because of slicing with m_tokens_len.
        Thus, it is reasonable to let the transformer output 513-dim for prediction
          rather than 514-dim (including the padding flag).
    '''
    if sem_token is not None:
        parts_cls_pred = trans_encoder(input_parts, feat_clip_text, sem_token_len)
        sem_cls_pred = parts_cls_pred[0][:, :semantic_len_max, :].contiguous()
        parts_cls_pred = [parts_cls_pred[i][:, semantic_len_max:, :] for i in range(len(parts_cls_pred))]
    else:
        parts_cls_pred = trans_encoder(input_parts, feat_clip_text)
    for i in range(len(parts_cls_pred)):
        parts_cls_pred[i] = parts_cls_pred[i].contiguous()


    '''forward time'''
    forward_time = time.time() - start_time
    start_time = time.time()


    parts_loss_cls = {
        'Root': 0.0,
        'R_Leg': 0.0,
        'L_Leg': 0.0,
        'Backbone': 0.0,
        'R_Arm': 0.0,
        'L_Arm': 0.0,
    }


    '''
    Parallel loss and acc computation.
    '''
    for j, name in enumerate(['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

        '''
        loss_ce input:
            parts_cls_pred[j]: (B, m_tokens_len + 1, 513), 
            gt_parts[j]:   (B, m_tokens_len + 1),      always including the end flag, even though input may not include.
            m_tokens_len:  (B,)
            max of m_token_len: 50
        '''

        batch_loss = loss_ce_batch(parts_cls_pred[j].permute(0,2,1), gt_parts[j])

        # re-weight the loss to be same to T2M-GPT
        num = (batch_loss != 0).sum(dim=1)
        weight = gt_parts[j].shape[1] / (num + 1e-6)
        parts_loss_cls[name] = (batch_loss.mean(dim=1) * weight).mean()

        # Accuracy
        with torch.no_grad():

            # predict the class according to the max logits
            _, cls_pred_index_determine = torch.max(parts_cls_pred[j], dim=2)  # cls_pred_index (64, 51)

            # predict the class according to their probability distribution
            # probs = torch.softmax(parts_cls_pred[j], dim=2)
            # predict the class according to their probability distribution
            # 添加数值稳定化处理
            logits = parts_cls_pred[j]
            # logits = logits - logits.max(dim=-1, keepdim=True)[0]  # 减去最大值以提高数值稳定性
            probs = torch.softmax(logits, dim=2)
            
            # 添加安全检查，将NaN值替换为均匀分布
            # if torch.isnan(probs).any():
            #     probs = torch.ones_like(probs) / probs.size(-1)
            dist = Categorical(probs)
            cls_pred_index_sample = dist.sample()

            right_num_determine[name] += (cls_pred_index_determine == gt_parts[j]).sum().item()
            right_num_sample[name] += (cls_pred_index_sample == gt_parts[j]).sum().item()

    if sem_token is not None:
        sem_loss = loss_ce_batch(sem_cls_pred.permute(0,2,1), sem_token)
        num_sem = (sem_loss != 0).sum(dim=1)
        weight_sem = sem_token.shape[1] / (num_sem + 1e-6)
        parts_loss_cls['sem'] = (sem_loss.mean(dim=1) * weight_sem).mean()
        with torch.no_grad():
            _, cls_pred_index_determine_sem = torch.max(sem_cls_pred, dim=2)  # cls_pred_index (64, 51)
            # probs = torch.softmax(sem_cls_pred, dim=2)
            # 对semantic部分也做同样的处理
            sem_logits = sem_cls_pred
            # sem_logits = sem_logits - sem_logits.max(dim=-1, keepdim=True)[0]
            sem_probs = torch.softmax(sem_logits, dim=2)
            
            # if torch.isnan(sem_probs).any():
            #     sem_probs = torch.ones_like(sem_probs) / sem_probs.size(-1)
            dist_sem = Categorical(sem_probs)
            cls_pred_index_sample_sem = dist_sem.sample()
            right_num_determine['sem'] += (cls_pred_index_determine_sem == sem_token).sum().item()
            right_num_sample['sem'] += (cls_pred_index_sample_sem == sem_token).sum().item()



    ## global loss
    loss_cls = 0.0
    for name in ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']:
        loss_cls = loss_cls + parts_loss_cls[name]
    if sem_token is not None:
        loss_cls = loss_cls + parts_loss_cls['sem']


    '''loss compute time'''
    loss_compute_time = time.time() - start_time
    start_time = time.time()


    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    '''backward time'''
    backward_time = time.time() - start_time


    for name in ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']:
        avg_loss_cls[name] = avg_loss_cls[name] + parts_loss_cls[name].item()
        
    if sem_token is not None:
        avg_loss_cls['sem'] = avg_loss_cls['sem'] + parts_loss_cls['sem'].item()

    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()
    if sem_token is not None:
        nb_sample_sem_train = nb_sample_sem_train + (sem_token_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter == 0:
        msg = f"Train. Iter {nb_iter}:"
        for name in ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']:

            avg_loss_cls[name] = avg_loss_cls[name] / args.print_iter
            avg_acc_determine[name] = right_num_determine[name] * 100 / nb_sample_train
            avg_acc_sample[name] = right_num_sample[name] * 100 / nb_sample_train
            writer.add_scalar('./Loss/train_' + name, avg_loss_cls[name], nb_iter)
            writer.add_scalar('./ACC_determine/train_' + name, avg_acc_determine[name], nb_iter)
            writer.add_scalar('./ACC_sample/train_' + name, avg_acc_sample[name], nb_iter)
            msg += f" [{name}] Loss. {avg_loss_cls[name]:.4f}, ACC_deter. {avg_acc_determine[name]:.3f}, ACC_sample. {avg_acc_sample[name]:.3f}"
            avg_loss_cls[name] = 0.
            right_num_determine[name] = 0
            right_num_sample[name] = 0
        if sem_token is not None:
            avg_loss_cls['sem'] = avg_loss_cls['sem'] / args.print_iter
            avg_acc_determine['sem'] = right_num_determine['sem'] * 100 / nb_sample_sem_train
            avg_acc_sample['sem'] = right_num_sample['sem'] * 100 / nb_sample_sem_train
            writer.add_scalar('./Loss/train_sem', avg_loss_cls['sem'], nb_iter)
            writer.add_scalar('./ACC_determine/train_sem', avg_acc_determine['sem'], nb_iter)
            writer.add_scalar('./ACC_sample/train_sem', avg_acc_sample['sem'], nb_iter)
            msg += f" [sem] Loss. {avg_loss_cls['sem']:.4f}, ACC_deter. {avg_acc_determine['sem']:.3f}, ACC_sample. {avg_acc_sample['sem']:.3f}"
            avg_loss_cls['sem'] = 0
            right_num_determine['sem'] = 0
            right_num_sample['sem'] = 0

        logger.info(msg)
        nb_sample_train = 0
        nb_sample_sem_train = 0
    if nb_iter % args.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
            eval_bodypart.evaluation_transformer_batch(
                args.run_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
                best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
                clip_model=clip_model, eval_wrapper=eval_wrapper, semantic_flag=(vqvae_train_args.lgvq>0), draw=False)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break