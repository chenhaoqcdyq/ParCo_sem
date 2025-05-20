import json
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import clip
from functools import partial
from transformers import AutoTokenizer
from typing import List, Dict
from transformers import BertTokenizer, BertModel

class VQMotionDatasetBodyPart(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4, print_warning=False, strategy='basic'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.text_token_dir = pjoin(self.data_root, 'texts_clip_token')
            self.text_feature_dir = pjoin(self.data_root, 'texts_clip_feature')
            # self.text_mask_dir = pjoin(self.data_root, 'texts_mask_deepseek')
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
        self.text_mask_dir =  pjoin(self.data_root, 'texts_mask_deepseek_processed')
        os.makedirs(self.text_token_dir, exist_ok=True)
        os.makedirs(self.text_feature_dir, exist_ok=True)
        from transformers import CLIPModel, CLIPTokenizer
        clip_model, preprocess = clip.load('ViT-B/32')
        joints_num = self.joints_num
        
        
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

        
        name_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    if print_warning:
                        print('Skip the motion:', name, '. motion length shorter than window_size')
                    continue
                text_data = []
                text_path = pjoin(self.text_dir, name + '.txt')
                text_tokens_path = pjoin(self.text_token_dir, name + '.pth')
                # if os.path.exists(text_path):
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        if len(line_split) > 0:
                            caption = line_split[0]
                            text_data.append(caption)
                # text_mask_path = pjoin(self.text_mask_dir, name + '.json')
                # text = ' '.join(text_data)  # 合并所有caption
                device = clip_model.text_projection.device
                self.lengths.append(motion.shape[0] - self.window_size)
                if not os.path.exists(text_tokens_path):
                    text_tokens = clip.tokenize(text_data, truncate = True).to(device)
                    torch.save(text_tokens, text_tokens_path)
                else:
                    text_tokens = torch.load(text_tokens_path, map_location=torch.device('cpu'))
                text_feature_path = pjoin(self.text_feature_dir, name + '.pth')
                if not os.path.exists(text_feature_path):
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text_tokens)
                    torch.save(text_features, text_feature_path)
                    
                else:
                    text_features = torch.load(text_feature_path, map_location=torch.device('cpu'))
                name_list.append(name)
                self.data.append({
                'motion': motion,      # 保持原始运动数据格式
                'text': text_data,           # 新增文本信息
                'text_token':text_tokens.cpu(),
                'text_feature': text_features.cpu(),
                'text_feature_all': text_features.cpu(),
                # 'text_mask':text_mask_data,
                'text_id': name
                })

            except:
                # Some motion may not exist in KIT dataset
                print('Unable to load:', name)

        print("Total number of motions {}".format(len(self.data)))
        self.tokenizer_name = "bert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_model.cuda()
        # self.bert_model.eval()  # 冻结BERT参数
        mask_list = [os.path.join(self.text_mask_dir, name + '.json') for name in name_list]
        self.raw_samples = self._load_raw_data(mask_list)
        self.masker = DynamicMaskGenerator()
        self.strategy = strategy
        self.static_labels = []
        # 生成动态标签
        for i in tqdm(range(len(self.raw_samples))):
            tmp = []
            for j in range(len(self.raw_samples[i])):
                tmp.append(self._get_static_labels(self.raw_samples[i][j]))
            self.static_labels.append(tmp)

    def _load_raw_data(self, file_paths: List[str]) -> List[Dict]:
        """加载原始未掩码数据"""
        samples = []
        
        for path in file_paths:
            with open(path, 'r') as f:
                data = json.load(f)
                for i in range(len(data['samples'])):
                    data['samples'][i]['name'] = path 
                samples.append(data['samples'])
        return samples


    def _get_static_labels(self, sample: Dict) -> List[Dict]:
        """从原始标注生成静态标签"""
        labels = []
        text = sample['original_text'].split()
        # 定义增强版位置查找（处理子词近似匹配）
        def find_all_positions(word: str, tokens: List[str]) -> List[int]:
            positions = []
            for i, token in enumerate(tokens):
                # 处理带标点的单词（如"jumping!" → "jumping"和"!"）
                clean_token = re.sub(r'[^\w]', '', token)
                if clean_token == word:
                    positions.append(i)
            return positions
        for action in sample['masked_word']:
            if len(action) > 2:
                print(action)
            # 处理核心动作
            core_words = action[0][0].split() if (action!= [] and len(action)>1 and action[0]!= [] and action[0][0]) else []
            for word in core_words:
                if word in text:
                    labels.append({
                        'type': 'core',
                        'word': word,
                        'position': text.index(word)
                    })
            # 处理身体部位
            body_words = action[0][1].split() if (action!= [] and len(action[0]) > 1 and action[0][1] != [] and action[0][1]) else []
            for word in body_words:
                if word in text:
                    labels.append({
                        'type': 'body',
                        'word': word,
                        'position': text.index(word)
                    })
            # 处理方向修饰
            if len(action) > 1 and action[1] != []:
                for i in range(len(action[1])):
                    dir_words = action[1][i].split() if action[1][i] else []
                    for word in dir_words:
                        if word in text:
                            labels.append({
                                'type': 'dir',
                                'word': word,
                                'position': text.index(word)
                            })
        # 处理新增的 direction_word 字段
        if 'direction_word' in sample and sample['direction_word']:
            for direction_item in sample['direction_word']:
                # 解析方向词格式 "direction:bodypart"，例如 "right:his hands"
                if ':' in direction_item:
                    direction, phase = direction_item.split(':', 1)
                    # 处理方向词
                    direction_words = direction.split()
                    for word in direction_words:
                        for pos in find_all_positions(word, text):  
                            if pos+1 < len(text) and phase == word+' '+text[pos+1]:
                                    labels.append({
                                        'type': 'bodypart_direction',
                                        'word': word,
                                        'position': pos
                                    })

        return labels

    def _build_labels(self, encoded: Dict, mask_labels: List) -> torch.Tensor:
        """构建动态标签张量"""
        labels = torch.full_like(encoded['input_ids'], -100)
        tokenized_text = self.bert_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        for label in mask_labels:
            pos = label['position']
            if pos < len(tokenized_text):
                # 处理subword情况
                if tokenized_text[pos].startswith('##'):
                    labels[0][pos] = -100
                else:
                    labels[0][pos] = self.bert_tokenizer.convert_tokens_to_ids(label['word'])
        return labels.squeeze()

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

    def whole2parts(self, motion, mode='t2m'):
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(motion, mode, window_size=self.window_size)
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    def get_each_part_vel(self, parts, mode='t2m'):
        parts_vel_list = get_each_part_vel(parts, mode=mode)
        return parts_vel_list

    def __len__(self):
        return len(self.data)
    
    def set_stage(self, stage):
        self.strategy = stage
    
    def __getitem__(self, item):

        # motion = self.data[item]
        data = self.data[item]
        motion = data['motion']
        text_list = data['text']
        text_tokens = data['text_token'].cpu()
        text_features = data['text_feature'].cpu()
        text_feature_alls = data['text_feature_all'].cpu()
        text_id = data['text_id']
        # text_mask = data['text_mask']
        
        text_random_id = random.randint(0, len(text_list) - 1)
        text = text_list[text_random_id]
        text_token = text_tokens[text_random_id]
        text_feature = text_features[text_random_id]
        text_feature_all = text_feature_alls[text_random_id]
        # text_mask = text_mask["samples"][text_random_id]
        # if text_feature.shape[0] != 512:
        #     print('text_feature:', text_feature)
        if len(text_list) != text_tokens.shape[0]:
            print('text_feature:', text_tokens)

        # Preprocess. We should set the slice of motion at getitem stage, not in the initialization.
        # If in the initialization, the augmentation of motion slice will be fixed, which will damage the diversity.
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        parts = self.whole2parts(motion, mode=self.dataset_name)

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts  # explicit written code for readability
        
        raw_sample_list = self.raw_samples[item]
        idx_sample = text_random_id
        if len(text_list) != len(raw_sample_list):
            print('text_list:', text_list)
        raw_sample = raw_sample_list[idx_sample]
        text = raw_sample['original_text']
        static_labels = self.static_labels[item][idx_sample]
        # 选择掩码策略
        if self.strategy == "progressive":
            stage = random.choice(self.masker.progressive_stages)
        else:
            stage = self.strategy
        
        # 动态生成掩码
        masked_data = self.masker.generate_masks(text, static_labels, stage)
        
        # 编码处理
        encoded = self.bert_tokenizer(
            masked_data['masked_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # with torch.no_grad():
        #     bert_outputs = self.bert_model(
        #         input_ids=encoded['input_ids'],
        #         attention_mask=encoded['attention_mask']
        #     )
        # 构建动态标签
        labels = self._build_labels(encoded, masked_data['labels'])
        text_mask = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            # 'bert_features': bert_outputs['last_hidden_state'].squeeze(),
            # 'bert_features_pool': bert_outputs['pooler_output'].squeeze(),
            'labels': labels
        }
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm], text, text_token, text_feature, text_feature_all, text_id, text_mask

class DynamicMaskGenerator:
    def __init__(self, 
                 mask_prob: float = 0.15,
                 progressive_stages: List[str] = ['basic', 'medium', 'advanced']):
        self.strategies = {
            'basic': partial(self._mask_core_dir, core_prob=0.5, dir_prob=0.3),
            'medium': partial(self._mask_core_body_dir, core_prob=0.6, dir_prob=0.4, body_prob=0.2),
            'advanced': self._full_mask,
            'random': self._random_mask
        }
        self.mask_token = "[MASK]"
        self.progressive_stages = progressive_stages
        
    def generate_masks(self, text: str, labels: List, strategy: str = 'random') -> Dict:
        """动态生成掩码"""
        words = text.split()
        masked = words.copy()
        new_labels = []
        
        if strategy in self.strategies:
            masked, new_labels = self.strategies[strategy](words, labels)
        else:
            masked, new_labels = self._random_mask(words, labels)
        
        return {
            'masked_text': ' '.join(masked),
            'labels': new_labels
        }

    def _mask_core(self, words, labels, prob=0.3):
        """核心动作掩码"""
        masked = words.copy()
        new_labels = []
        for word_info in labels:
            if word_info['type'] == 'core' and random.random() < prob:
                pos = word_info['position']
                try:
                    masked[pos] = self.mask_token
                except:
                    print('pos:', pos)
                    print('len(masked):', len(masked))
                    print('masked:', masked)
                new_labels.append(word_info)
        return masked, new_labels

    def _mask_core_dir(self, words, labels, core_prob=0.5, dir_prob=0.3):
        """核心+方向联合掩码"""
        masked, new_labels = self._mask_core(words, labels, core_prob)
        for word_info in labels:
            if (word_info['type'] == 'dir' or word_info['type'] == 'bodypart_direction') and random.random() < dir_prob:
                pos = word_info['position']
                try:
                    # masked[pos] = self.mask_token
                    masked[pos] = len(self.bert_tokenizer.tokenize(word_info['word']))
                except:
                    print('pos:', pos)
                    print('len(masked):', len(masked))
                    print('masked:', masked)
                new_labels.append(word_info)
        return masked, new_labels

    def _mask_core_body_dir(self, words, labels, core_prob=0.5, dir_prob=0.3, body_prob=0.3):
        """核心+方向联合掩码"""
        masked, new_labels = self._mask_core_dir(words, labels, core_prob, dir_prob)
        for word_info in labels:
            if word_info['type'] == 'body' and random.random() < body_prob:
                pos = word_info['position']
                masked[pos] = self.mask_token
                new_labels.append(word_info)
        return masked, new_labels

    def _full_mask(self, words, labels):
        """全要素渐进式掩码"""
        return self._mask_core_body_dir(words, labels, 0.8, 0.7, 0.5)

    def _random_mask(self, words, labels):
        """随机掩码策略"""
        masked = words.copy()
        new_labels = []
        for word_info in labels:
            if random.random() < 0.3:  # 基础概率
                pos = word_info['position']
                masked[pos] = self.mask_token
                new_labels.append(word_info)
        return masked, new_labels
    

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

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDatasetBodyPart(dataset_name, window_size=window_size, unit_length=unit_length)
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
