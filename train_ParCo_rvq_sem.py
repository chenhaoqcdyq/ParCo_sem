import os
import json
import re
# import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import  dataset_TM_eval_bodypart

# from dataset import dataset_VQ_bodypart_text_woclip, dataset_TM_eval_bodypart
from models import rvqvae_bodypart as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper

from options.get_eval_option import get_opt
import options.option_vq_bodypart as option_vq
from options.option_vq_bodypart import vqvae_bodypart_cfg, vqvae_bodypart_cfg_plus

import utils.losses as losses
import utils.utils_model as utils_model
import utils.eval_bodypart as eval_bodypart
from utils.word_vectorizer import WordVectorizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# warnings.filterwarnings('ignore')


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def freeze_encdec(net):
    for name, param in net.named_parameters():
        if ('dual' in name or 'lgvq' in name) and "bert_model" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, module in net.named_modules():
        if ('dual' in name or 'lgvq' in name) and "bert_model" not in name:
            module.train()
        else:
            module.eval()
    return net
# 在解析args后添加分布式初始化逻辑
def setup_distributed():
    # 读取环境变量
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def get_dataloader(dataset, batch_size, shuffle):
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size // world_size,  # 调整实际batch size
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True, 
        persistent_workers=True
    )
    return loader

# 确保所有进程使用相同种子
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reduce_metric(tensor):
    """将各GPU的指标张量求和后取平均"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / dist.get_world_size()

os.environ["MASTER_ADDR"]='127.0.0.1'
os.environ["MASTER_PORT"]='8675'
##### ---- Parse args ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
# args.vqvae_arch_cfg = vqvae_bodypart_cfg[args.vqvae_cfg]
if args.bodyconfig == True:
    args.vqvae_arch_cfg = vqvae_bodypart_cfg_plus[args.vqvae_cfg]
else:
    args.vqvae_arch_cfg = vqvae_bodypart_cfg[args.vqvae_cfg]
if args.lgvq > 0:
    from dataset import dataset_VQ_bodypart_text_mask_196 as dataset_VQ_bodypart_text
else:
    from dataset import dataset_VQ_bodypart_text_mask as dataset_VQ_bodypart_text
##### ---- Exp dirs ---- #####
"""
Directory of our exp:
./output  (arg.out_dir)
 ├── 00000-DATASET  (exp_number + dataset_name)
 │   └── VQVAE-EXP_NAME-DESC  (VQVAE + args.exp_name + desc)
 │       ├── events.out.XXX
 │       ├── net_best_XXX.pth
 │       ...
 │       ├── run.log
 │       ├── test_vqvae
 │       │   ├── ...
 │       │   ...
 │       ├── 0000-Trans-EXP_NAME-DESC  (stage2_exp_number + Trans + args.exp_name + desc)
 │       │   ├── quantized_dataset  (The quantized motion using VQVAE)
 │       │   ├── events.out.XXX
 │       │   ├── net_best_XXX.pth
 │       │   ...
 │       │   ├── run.log
 │       │   └── test_trans
 │       │       ├── ...
 │       │       ...
 │       ├── 0001-Trans-EXP_NAME-DESC
 │       ...
 ├── 00001-DATASET  (exp_number + dataset_name)
 ...
"""
# [Prepare description]
desc = args.dataname  # dataset
desc += f'-{args.vqvae_cfg}'

# Pick output directory.
prev_run_dirs = []

outdir = args.out_dir
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataname}-{args.exp_name}', f'VQVAE-{args.exp_name}-{desc}')
assert not os.path.exists(args.run_dir)
print('Creating output directory...')
os.makedirs(args.run_dir)
##### ---- ddp ---- #####
set_seed(args.seed)
if args.ddp:
    rank, local_rank, world_size = setup_distributed()
else:
    rank, local_rank, world_size = 0, 0, 1
device = torch.device(f'cuda:{local_rank}')
# else:
    # device = torch.device('cuda')
    # rank, local_rank, world_size = 0, 0, 1
args.rank = rank
args.local_rank = local_rank
args.world_size = world_size
##### ---- Logger ---- #####
logger = utils_model.get_logger(args.run_dir)
writer = SummaryWriter(args.run_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# save the training config
args.args_save_dir = os.path.join(args.run_dir, 'train_config.json')
args_dict = vars(args)
with open(args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)


w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
print('\n\n===> Constructing network...')
net = getattr(vqvae, f'HumanVQVAETransformerV{args.vision}')(args,  # use args to define different parameters in different quantizers
        parts_code_nb=args.vqvae_arch_cfg['parts_code_nb'],
        parts_code_dim=args.vqvae_arch_cfg['parts_code_dim'],
        parts_output_dim=args.vqvae_arch_cfg['parts_output_dim'],
        parts_hidden_dim=args.vqvae_arch_cfg['parts_hidden_dim'],
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm
    )    
if args.ddp:
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, bucket_cap_mb=200)                      
if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    if args.ddp:
        ckpt = torch.load(args.resume_pth, map_location=f'cuda:{local_rank}')
        net.module.load_state_dict(ckpt['net'], strict=False)
    else:
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        if args.lgvq == 5:
            try:
                new_ckpt = {}
                for k, v in ckpt['net'].items():
                    if 'bert_model' not in k:
                        new_ckpt[k] = v
                net.load_state_dict(new_ckpt, strict=False)
                print("load lgvq model")
            except:
                new_ckpt = {}
                for k, v in ckpt['net'].items():
                    if 'lgvq' not in k:
                        new_ckpt[k] = v
                net.load_state_dict(new_ckpt, strict=False)
                print("load wo lgvq model")
        else:
            net.load_state_dict(ckpt['net'], strict=False)

net.train()       
if args.freeze_encdec:
    net = freeze_encdec(net)
# net.eval()
net.to(device)

##### ---- Dataloader ---- #####
print('\n\n===> Constructing dataset and dataloader...\n\n')
train_loader = dataset_VQ_bodypart_text.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)
if args.ddp:
    train_loader = get_dataloader(train_loader.dataset, args.batch_size, True)
train_loader_iter = dataset_VQ_bodypart_text.cycle(train_loader)

    
val_loader = dataset_TM_eval_bodypart.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)
if args.ddp:
    val_loader = get_dataloader(val_loader.dataset, 32, False)
##### ---- Optimizer & Scheduler ---- #####
print('\n===> Constructing optimizer, scheduler, and Loss...')
def get_optimizer_params(net, base_lr, vqvae_lr):
    params = []
    for name, param in net.named_parameters():
        if param.requires_grad:  # 只添加requires_grad=True的参数
            if 'vqvae' in name and 'enhancedvqvae' not in name:
                params.append({'params': param, 'lr': vqvae_lr})
            else:
                params.append({'params': param, 'lr': base_lr})
    return params

# optimizer_params = get_optimizer_params(net, args.lr, args.lr * 0.1)
optimizer = optim.AdamW(get_optimizer_params(net, args.lr, args.lr), betas=(0.9, 0.99), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
Loss = losses.ReConsLossBodyPart(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
print('\n===> Start warm-up training\n\n')

avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
avg_contrastive, avg_disentangle = 0., 0.
for nb_iter in range(1, args.warm_up_iter):
    if args.ddp: 
        train_loader.sampler.set_epoch(nb_iter)
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    # train_loader_iter = dataset_VQ_bodypart_text.cycle(train_loader)
    batch = next(train_loader_iter)
    if len(batch) == 3:
        gt_parts, text, text_id = batch
    elif len(batch) == 6:   
        gt_parts, text, text_token, text_feature, text_feature_all, text_id = batch
    elif len(batch) == 7:
        gt_parts, text, text_token, text_feature, text_feature_all, text_id, text_mask = batch
    elif len(batch) == 8:
        gt_parts, text, text_token, text_feature, text_feature_all, text_id, text_mask, motion_mask = batch
    else:
        raise ValueError("The length of batch is not correct.")
    for i in range(len(gt_parts)):
        gt_parts[i] = gt_parts[i].to(device).float()

    if args.lgvq > 0:
        cond = [text_feature, text_id, text_mask, motion_mask]
        # print(args.vision, "cond = [text_feature, text_id, text_mask, motion_mask]")
    elif args.vision >= 17:
        cond = [text_feature, text_id]
        motion_mask = None
        # print(args.vision, "cond = [text_feature, text_id]")
    else:
        cond = text
        motion_mask = None
        # print(args.vision, "cond = text")
    pred_parts, loss_commit_list, perplexity_list, loss_extend = net(gt_parts, cond)
    # contrastive_loss = loss_extend['contrastive']
    if isinstance(loss_extend, list):
        contrastive_loss = loss_extend[0]
        if isinstance(loss_extend[1], list):
            disentangle_loss = losses.gather_loss_list(loss_extend[1])
        else:
            disentangle_loss = loss_extend[1]
    else:
        contrastive_loss = loss_extend
        disentangle_loss = torch.tensor(0.0).to(device)
    pred_parts_vel = dataset_VQ_bodypart_text.get_each_part_vel(
        pred_parts, mode=args.dataname)
    gt_parts_vel = dataset_VQ_bodypart_text.get_each_part_vel(
        gt_parts, mode=args.dataname)

    loss_motion_list = Loss(pred_parts, gt_parts, motion_mask)  # parts motion reconstruction loss
    loss_vel_list = Loss.forward_vel(pred_parts_vel, gt_parts_vel, motion_mask)  # parts velocity recon loss

    loss_motion = losses.gather_loss_list(loss_motion_list)
    loss_commit = losses.gather_loss_list(loss_commit_list)
    loss_vel = losses.gather_loss_list(loss_vel_list)

    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel + contrastive_loss * args.contrastive + disentangle_loss * args.Disentangle


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    perplexity = losses.gather_loss_list(perplexity_list)
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_contrastive += contrastive_loss.item()
    avg_disentangle += disentangle_loss.item()
    
    if nb_iter % args.print_iter == 0 and rank == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_contrastive /= args.print_iter
        avg_disentangle /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        logger.info(f"Contrastive loss: {avg_contrastive:.5f} \t Disentangle loss: {avg_disentangle:.5f}")
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        avg_contrastive, avg_disentangle = 0., 0.

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe = \
    eval_bodypart.evaluation_vqvae(
        args.run_dir, val_loader, net, logger, writer, 0,
        best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
        eval_wrapper=eval_wrapper)
if args.freeze_encdec:
    net = freeze_encdec(net)

##### ---- Training ---- #####

print('\n\n===> Start training\n\n')
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
avg_contrastive, avg_disentangle = 0., 0.

for nb_iter in range(1, args.total_iter + 1):
    if args.ddp: 
        train_loader.sampler.set_epoch(nb_iter)  # 确保shuffle正确
    # if args.ddp: train_loader.sampler.set_epoch(nb_iter)
    batch = next(train_loader_iter)
    if len(batch) == 3:
        gt_parts, text, text_id = batch
    elif len(batch) == 6:   
        gt_parts, text, text_token, text_feature, text_feature_all, text_id = batch
    elif len(batch) == 7:
        gt_parts, text, text_token, text_feature, text_feature_all, text_id, text_mask = batch
    elif len(batch) == 8:
        gt_parts, text, text_token, text_feature, text_feature_all, text_id, text_mask, motion_mask = batch
        # motion_mask = motion_mask.to(gt_parts[0].device).bool()
    else:
        raise ValueError("The length of batch is not correct.")
    for i in range(len(gt_parts)):
        gt_parts[i] = gt_parts[i].to(device).float()
    if args.lgvq > 0:
        cond = [text_feature, text_id, text_mask, motion_mask]
        if nb_iter > args.sem_iter and train_loader.dataset.strategy == 'basic' :
            train_loader.dataset.strategy = 'medium'
            train_loader_iter = dataset_VQ_bodypart_text.cycle(train_loader)
            
    elif args.vision >= 17:
        cond = [text_feature, text_id]
        motion_mask = None
    else:
        cond = text
        motion_mask = None
    pred_parts, loss_commit_list, perplexity_list, loss_extend = net(gt_parts, cond)
    if isinstance(loss_extend, list):
        contrastive_loss = loss_extend[0]
        if isinstance(loss_extend[1], list):
            disentangle_loss = losses.gather_loss_list(loss_extend[1])
        else:
            disentangle_loss = loss_extend[1]
    else:
        contrastive_loss = loss_extend
        disentangle_loss = torch.tensor(0.0).to(device)
    pred_parts_vel = dataset_VQ_bodypart_text.get_each_part_vel(
        pred_parts, mode=args.dataname)
    gt_parts_vel = dataset_VQ_bodypart_text.get_each_part_vel(
        gt_parts, mode=args.dataname)

    loss_motion_list = Loss(pred_parts, gt_parts, motion_mask)  # parts motion reconstruction loss
    loss_vel_list = Loss.forward_vel(pred_parts_vel, gt_parts_vel, motion_mask)  # parts velocity recon loss


    loss_motion = losses.gather_loss_list(loss_motion_list)
    loss_commit = losses.gather_loss_list(loss_commit_list)
    loss_vel = losses.gather_loss_list(loss_vel_list)

    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel + contrastive_loss * args.contrastive  + disentangle_loss * args.Disentangle
    
    optimizer.zero_grad()
    loss.backward()
    if args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    perplexity = losses.gather_loss_list(perplexity_list)
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_contrastive += contrastive_loss.item()
    avg_disentangle += disentangle_loss.item()
    
    if nb_iter % args.print_iter == 0 and rank == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_contrastive /= args.print_iter
        avg_disentangle /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        writer.add_scalar('./Train/Contrastive', avg_contrastive, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        logger.info(f"Contrastive loss: {avg_contrastive:.5f} \t Disentangle loss: {avg_disentangle:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,
        avg_contrastive, avg_disentangle = 0., 0.

    if nb_iter % args.eval_iter == 0 and rank == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe = \
            eval_bodypart.evaluation_vqvae(
                args.run_dir, val_loader, net, logger, writer, nb_iter,
                best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
                eval_wrapper=eval_wrapper, best_mpjpe=best_mpjpe)
        if args.freeze_encdec:
            net = freeze_encdec(net)