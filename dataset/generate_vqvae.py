import os 
from os.path import join as pjoin
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from dataset import dataset_TM_train_bodypart
from dataset import dataset_TM_eval_bodypart
from dataset import dataset_tokenize_bodypart
import codecs as cs

def get_text(name):
    text_dir = 'dataset/HumanML3D/texts'
    text_data = []
    with cs.open(pjoin(text_dir, name + '.txt')) as f:
        for line in f.readlines():
            text_dict = {}
            line_split = line.strip().split('#')
            caption = line_split[0]
            text_data.append(caption)
    return text_data
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
net = vqvae_bodypart.HumanVQVAEBodyPart(
    vqvae_train_args,  # use args to define different parameters in different quantizers
    vqvae_train_args['vqvae_arch_cfg']['parts_code_nb'],
    vqvae_train_args['vqvae_arch_cfg']['parts_code_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_output_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_hidden_dim'],
    vqvae_train_args['down_t'],
    vqvae_train_args['stride_t'],
    vqvae_train_args['depth'],
    vqvae_train_args['dilation_growth_rate'],
    vqvae_train_args['vq_act'],
    vqvae_train_args['vq_norm']
)
print('===> Loading VQVAE checkpoint from {}...'.format(args.vqvae_ckpt_path))
ckpt = torch.load(args.vqvae_ckpt_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()


##### ---- Prepare tokenized motion dataset ---- #####
print('\n\n===> Preparing the quantized motion for training:')

if not args.use_existing_vq_data:
    # [token loader]
    print('===> Constructing dataset for tokenize...')
    train_loader_token = dataset_tokenize_bodypart.DATALoader(
        args.dataname, 1, unit_length=2**args.down_t, val=True)

    # Get code (quantized motion)
    print('===> Getting the code...', args.vq_dir)
    for batch in tqdm(train_loader_token):
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
        text = get_text(name[0])
        motion_part = []
        for i, part_name in enumerate(['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):
            tok_p = tokenized_parts[i]  # (B, nframes)
            tok_p = tok_p.cpu().numpy()
            motion_part.append(tok_p)
        # use name[0] because the batch size is 1
        # np.save(pjoin(args.vq_dir, name[0] + '_' + part_name + '.npy'), tok_p)
        np.savez(pjoin(args.vq_dir, name[0] + '.npz'), motion=motion_part, text=text, name=name[0])
else:
    print('\n\n')
    print('####################################################')
    print('==> Warning: using existing tokenized motion data!  ')
    print('==> It is used for debug, do not use it in training!')
    print('####################################################')
    print('\n\n')