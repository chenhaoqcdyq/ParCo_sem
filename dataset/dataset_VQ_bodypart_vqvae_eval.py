import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import clip
from models import rvqvae_bodypart as vqvae
from utils.motion_process import recover_from_ric
import visualize.plot_3d_global as plot_3d

class VQMotionDatasetBodyPart(data.Dataset):
    def __init__(self, args, dataset_name, window_size=64, unit_length=4, print_warning=False):
        self.args = args
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.max_length = 196

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.text_token_dir = pjoin(self.data_root, 'texts_clip_token')
            self.text_feature_dir = pjoin(self.data_root, 'texts_clip_feature')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/Decomp_SP001_SM001_H512/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.text_token_dir = pjoin(self.data_root, 'texts_clip_token')
            self.text_feature_dir = pjoin(self.data_root, 'texts_clip_feature')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/Decomp_SP001_SM001_H512/meta'
        os.makedirs(self.text_token_dir, exist_ok=True)
        os.makedirs(self.text_feature_dir, exist_ok=True)
        # from transformers import CLIPModel, CLIPTokenizer
        # self.text_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        # self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        # clip_model, preprocess = clip.load('ViT-B/32')
        # joints_num = self.joints_num
        self.vqvae = getattr(vqvae, f'HumanVQVAETransformerV{args.vision}')(args,  # use args to define different parameters in different quantizers
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
        if args.resume_pth:
            # logger.info('loading checkpoint from {}'.format(args.resume_pth))
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            self.vqvae.load_state_dict(ckpt['net'], strict=True)
        else:
            raise Exception('No checkpoint found!')
        self.vqvae.cuda()
        self.vqvae.eval()
        self.vqvae_dir = pjoin(self.data_root, f'V{args.vision}_vqvae')
        if self.vqvae_dir is not None:
            os.makedirs(self.vqvae_dir, exist_ok=True)
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        self.mean = mean
        self.std = std

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            # try:
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if motion.shape[0] < self.window_size:
                if print_warning:
                    print('Skip the motion:', name, '. motion length shorter than window_size')
                continue
            text_data = []
            text_path = pjoin(self.text_dir, name + '.txt')
            # text_tokens_path = pjoin(self.text_token_dir, name + '.pth')
            # if os.path.exists(text_path):
            with cs.open(text_path, 'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    if len(line_split) > 0:
                        caption = line_split[0]
                        text_data.append(caption)
            vis_vqvae_path = pjoin(self.vqvae_dir, name + '.gif')
            vis_gt_path = pjoin(self.vqvae_dir, name + '_gt.gif')
            motion_vqvae = (motion - self.mean) / self.std
            motion_vqvae = self.whole2parts(motion_vqvae, mode=self.dataset_name, window_size=motion.shape[0])
            for i in range(len(motion_vqvae)):
                motion_vqvae[i] = motion_vqvae[i].unsqueeze(0).cuda()
            with torch.no_grad():
                rec_data, _, _, _ = self.vqvae(motion_vqvae)
                pred_pose = parts2whole(rec_data)
                pred_xyz = recover_from_ric((pred_pose[:,:motion.shape[0],...]*torch.from_numpy(std).cuda() + torch.from_numpy(mean).cuda()).float(), 22)
                xyz = pred_xyz.reshape(1, -1, 22, 3)
                pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), [caption], [vis_vqvae_path])
                gt_xyz = recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
                gtxyz = gt_xyz.reshape(1, -1, 22, 3)
                plot_3d.draw_to_batch(gtxyz.detach().cpu().numpy(), [caption], [vis_gt_path])
            # text = ' '.join(text_data)  # 合并所有caption
            # device = clip_model.text_projection.device
            # self.lengths.append(motion.shape[0] - self.window_size)
            # if not os.path.exists(text_tokens_path):
            #     text_tokens = clip.tokenize(text_data, truncate = True).to(device)
            #     torch.save(text_tokens, text_tokens_path)
            # else:
            #     text_tokens = torch.load(text_tokens_path, map_location=torch.device('cpu'))
            # text_feature_path = pjoin(self.text_feature_dir, name + '.pth')
            # text_feature_all_path = pjoin(self.text_feature_dir, name + '_all.pth')
            # if not os.path.exists(text_feature_path):
            #     with torch.no_grad():
            #         # outputs = self.text_model(**inputs).last_hidden_state.mean(dim=1)
            #         text_features = clip_model.encode_text(text_tokens)
            #         text_allseq = clip_model.encode_text_allseq(text_tokens)
            #     # torch.save(text_tokens, text_tokens_path)
            #     torch.save(text_features, text_feature_path)
            #     torch.save(text_allseq, text_feature_all_path)
                
            # else:
            #     text_features = torch.load(text_feature_path, map_location=torch.device('cpu'))
            #     text_allseq = torch.load(text_feature_all_path, map_location=torch.device('cpu'))
                # text_tokens = torch.load(text_tokens_path)
                # if text_tokens.shape[0] != len(text_data):
                #     inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=77)
                #     with torch.no_grad():
                #         text_tokens = self.text_model.get_text_features(**inputs)
                #     torch.save(text_tokens, text_tokens_path)
                # if text_tokens.shape[0] == 1:
                #     text_tokens = text_tokens[0]
            self.data.append({
            'motion': motion,      # 保持原始运动数据格式
            'text': text_data,           # 新增文本信息
            # 'text_token':text_tokens.cpu(),
            # 'text_feature': text_features.cpu(),
            # 'text_feature_all': text_allseq.cpu(),
            'text_id': name
            })

            # except:
            #     # Some motion may not exist in KIT dataset
            #     print('Unable to load:', name)

        print("Total number of motions {}".format(len(self.data)))

    def parts2whole(self, parts, mode='t2m', shared_joint_rec_mode='Avg'):
        rec_data = parts2whole(parts, mode, shared_joint_rec_mode)
        return rec_data

    def inv_transform(self, data):
        # de-normalization
        return data * self.std + self.mean
    
    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def whole2parts(self, motion, mode='t2m', window_size=None):
        if window_size is None:
            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(motion, mode, window_size=self.window_size)
        else:
            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(motion, mode, window_size=window_size)
        if Root.shape[0] < self.max_length:
            nframes = Root.shape[0]
            Root = torch.cat([Root, torch.zeros((self.max_length - nframes, Root.shape[1])).to(Root.device)], dim=0)
            R_Leg = torch.cat([R_Leg, torch.zeros((self.max_length - nframes, R_Leg.shape[1])).to(R_Leg.device)], dim=0)
            L_Leg = torch.cat([L_Leg, torch.zeros((self.max_length - nframes, L_Leg.shape[1])).to(L_Leg.device)], dim=0)
            Backbone = torch.cat([Backbone, torch.zeros((self.max_length - nframes, Backbone.shape[1])).to(Backbone.device)], dim=0)
            R_Arm = torch.cat([R_Arm, torch.zeros((self.max_length - nframes, R_Arm.shape[1])).to(R_Arm.device)], dim=0)
            L_Arm = torch.cat([L_Arm, torch.zeros((self.max_length - nframes, L_Arm.shape[1])).to(L_Arm.device)], dim=0)
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    def get_each_part_vel(self, parts, mode='t2m'):
        parts_vel_list = get_each_part_vel(parts, mode=mode)
        return parts_vel_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        # motion = self.data[item]
        data = self.data[item]
        motion = data['motion']
        text_list = data['text']
        # text_tokens = data['text_token'].cpu()
        # text_features = data['text_feature'].cpu()
        # text_feature_alls = data['text_feature_all'].cpu()
        text_id = data['text_id']
        
        # text_random_id = random.randint(0, len(text_list) - 1)
        text = text_list
        text += [''] * (5 - len(text))
        # text_token = text_tokens[text_random_id]
        # text_feature = text_features[text_random_id]
        # text_feature_all = text_feature_alls[text_random_id]
        
        # # if text_feature.shape[0] != 512:
        # #     print('text_feature:', text_feature)
        # if len(text_list) != text_tokens.shape[0]:
        #     print('text_feature:', text_tokens)

        # Preprocess. We should set the slice of motion at getitem stage, not in the initialization.
        # If in the initialization, the augmentation of motion slice will be fixed, which will damage the diversity.
        # idx = random.randint(0, len(motion) - self.window_size)
        motion_len = len(motion) if len(motion) < self.max_length else self.max_length
        
        # motion = motion[idx:idx+self.window_size]
        motion = motion[:motion_len]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        parts = self.whole2parts(motion, mode=self.dataset_name, window_size=motion_len)

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts  # explicit written code for readability
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm], text, text_id, motion_len


def whole2parts(motion, mode='t2m', window_size=None):
    # motion
    if mode == 't2m':
        # 263-dims motion is actually an augmented motion representation
        # split the 263-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        aug_data = torch.from_numpy(motion)  # (nframes, 263)
        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(torch.int64)        # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(torch.int64)        # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)      # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)    # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)    # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat([root_data, local_vel[:,0,:]], dim=1)  # (nframes, 4+3=7)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)

        Root = root_data  # (nframes, 4+3=7)
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)  # (nframes, 4*12+2=50)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)  # (nframes, 4*12+2=50)
        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)

    elif mode == 'kit':
        # 251-dims motion is actually an augmented motion representation
        # split the 251-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        aug_data = torch.from_numpy(motion)  # (nframes, 251)
        joints_num = 21
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]

        # move the root joint 0-th out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(torch.int64)        # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(torch.int64)        # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(torch.int64)      # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)    # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)    # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat([root_data, local_vel[:,0,:]], dim=1)  # (nframes, 4+3=7)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)

        Root = root_data  # (nframes, 4+3=7)
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)  # (nframes, 4*12+2=50)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)  # (nframes, 4*12+2=50)
        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)

    else:
        raise Exception()

    return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]


def parts2whole(parts, mode='t2m', shared_joint_rec_mode='Avg'):
    assert isinstance(parts, list)

    if mode == 't2m':
        # Parts to whole. (7, 50, 50, 60, 60, 60) ==> 263
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 22
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(device, dtype=torch.int64)        # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(device, dtype=torch.int64)        # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(device, dtype=torch.int64)      # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(device, dtype=torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(device, dtype=torch.int64)  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[:,0,:] = Root[:,4:]

        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(bs, nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip([R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]):
            # rec_ric_data[:, idx - 1, :] = part[:, :, :3]
            # rec_rot_data[:, idx - 1, :] = part[:, :, 3:9]
            # rec_local_vel[:, idx, :] = part[:, :, 9:]

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 9th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 9

        if shared_joint_rec_mode == 'L_Arm':
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'R_Arm':
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'Backbone':
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == 'Avg':
            rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
            rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
            rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3

        else:
            raise Exception()

        # Concate them to 263-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=1)

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=2)

    elif mode == 'kit':

        # Parts to whole. (7, 62, 62, 48, 48, 48) ==> 251
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 21
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(device, dtype=torch.int64)  # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(device, dtype=torch.int64)  # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(device, dtype=torch.int64)            # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(device, dtype=torch.int64)          # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(device, dtype=torch.int64)         # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[:,0,:] = Root[:,4:]

        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(bs, nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip([R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]):

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 3-th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 3

        if shared_joint_rec_mode == 'L_Arm':
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'R_Arm':
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'Backbone':
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == 'Avg':
            rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
            rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
            rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3

        else:
            raise Exception()

        # Concate them to 251-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=1)

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=2)

    else:
        raise Exception()

    return rec_data


def get_each_part_vel(parts, mode='t2m'):
    assert isinstance(parts, list)

    if mode == 't2m':
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]

    elif mode == 'kit':
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]

    else:
        raise Exception()

    return parts_vel_list  # [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]


# def custom_collate_fn(batch):
#     motions, texts, text_features, text_ids = zip(*batch)
    
#     # 将 motions 转换为张量
#     motions = [torch.stack(parts) for parts in zip(*motions)]
    
#     # 将 texts 和 text_features 转换为张量
#     texts = list(texts)
#     text_features = torch.stack(text_features)
    
#     return motions, texts, text_features, text_ids

def DATALoader(args, dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDatasetBodyPart(args, dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                            #   num_workers=1,
                                            #   collate_fn=custom_collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
