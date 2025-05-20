import json
import os
import random
from pathlib import Path
from typing import List, Dict, Generator

class ProgressiveMasker:
    def __init__(self, config: Dict = None):
        self.default_config = {
            'stages': {
                'basic': {'core': 0.5, 'dir': 0.3, 'body': 0.0},
                'medium': {'core': 0.7, 'dir': 0.5, 'body': 0.3},
                'advanced': {'core': 0.8, 'dir': 0.7, 'body': 0.5}
            },
            'max_samples_per_file': 1000
        }
        self.config = config or self.default_config

    def process_dataset(self, input_dir: str, output_dir: str):
        """批量处理JSON文件"""
        os.makedirs(output_dir, exist_ok=True)
        input_path = Path(input_dir)
        for json_file in input_path.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
            print(f"Processing {json_file.name}...")
            # print(data)
            results = []
            for sample in self._process_samples(data['samples']):
                results.append(sample)
                if len(results) >= self.config['max_samples_per_file']:
                    self._save_batch(output_dir, json_file.name, results)
                    results = []
            
            if results:
                self._save_batch(output_dir, json_file.name, results)

    def _process_samples(self, samples: List) -> Generator[Dict, None, None]:
        """样本处理流水线"""
        for sample in samples:
            if not self._validate_sample(sample):
                continue
                
            original_words = set(sample['original_text'].lower().split())
            for stage in self.config['stages']:
                masked_text, labels = self._apply_mask(
                    sample['original_text'],
                    sample['masked_word'],
                    original_words,
                    self.config['stages'][stage]
                )
                yield {
                    'original': sample['original_text'],
                    'masked': masked_text,
                    'labels': labels,
                    'stage': stage
                }

    def _apply_mask(self, text: str, mask_word: List, vocab: set, prob: Dict) -> tuple:
        """应用单阶段掩码"""
        words = text.split()
        masked = words.copy()
        labels = []
        
        # 遍历每个时序动作组
        for action_group in mask_word:
            # 处理核心动作
            core_actions = [w for w in action_group[0] if w and w in vocab]
            if core_actions and random.random() < prob['core']:
                target = random.choice(core_actions)
                idx = words.index(target)
                masked[idx] = '[MASK]'
                labels.append({'type': 'core', 'word': target})
            
            # 处理方向修饰
            directions = [w for w in action_group[1] if w and w in vocab]
            if directions and random.random() < prob['dir']:
                target = random.choice(directions)
                idx = words.index(target)
                masked[idx] = '[MASK]'
                labels.append({'type': 'dir', 'word': target})
            
            # 处理身体部位（仅高级阶段）
            if prob['body'] > 0:
                body_parts = [w for w in action_group[0][1:] if w and w in vocab]
                if body_parts and random.random() < prob['body']:
                    target = random.choice(body_parts)
                    idx = words.index(target)
                    masked[idx] = '[MASK]'
                    labels.append({'type': 'body', 'word': target})
        
        return ' '.join(masked), labels

    def _validate_sample(self, sample: Dict) -> bool:
        """数据验证"""
        required_keys = {'original_text', 'masked_word'}
        if not all(k in sample for k in required_keys):
            return False
            
        original_words = sample['original_text'].split()
        for action in sample['masked_word']:
            if len(action) != 2 or len(action[0]) < 1:
                return False
            for word in action[0] + action[1]:
                if word and word not in original_words:
                    return False
        return True

    def _save_batch(self, output_dir: str, filename: str, data: List):
        """批量保存"""
        output_path = Path(output_dir) / f"masked_{filename}"
        with open(output_path, 'w') as f:
            json.dump({'samples': data}, f, indent=2)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
from typing import List, Dict
from functools import partial
from tqdm import tqdm

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
                masked[pos] = self.mask_token
                new_labels.append(word_info)
        return masked, new_labels

    def _mask_core_dir(self, words, labels, core_prob=0.5, dir_prob=0.3):
        """核心+方向联合掩码"""
        masked, new_labels = self._mask_core(words, labels, core_prob)
        for word_info in labels:
            if word_info['type'] == 'dir' and random.random() < dir_prob:
                pos = word_info['position']
                masked[pos] = self.mask_token
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

class DynamicMotionDataset(Dataset):
    def __init__(self, 
                 file_paths: List[str], 
                 tokenizer_name: str = "bert-base-uncased",
                 mask_strategy: str = "progressive"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.raw_samples = self._load_raw_data(file_paths)
        self.masker = DynamicMaskGenerator()
        self.strategy = mask_strategy
        self.static_labels = []
        # 生成动态标签
        for i in tqdm(range(len(self.raw_samples))):
            for j in range(len(self.raw_samples[i])):
                tmp = self._get_static_labels(self.raw_samples[i][j])
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

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx) -> Dict:
        raw_sample_list = self.raw_samples[idx]
        raw_sample = random.choice(raw_sample_list)
        text = raw_sample['original_text']
        static_labels = self.static_labels[idx]
        # 选择掩码策略
        if self.strategy == "progressive":
            stage = random.choice(self.masker.progressive_stages)
        else:
            stage = self.strategy
        
        # 动态生成掩码
        masked_data = self.masker.generate_masks(text, static_labels, stage)
        
        # 编码处理
        encoded = self.tokenizer(
            masked_data['masked_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # 构建动态标签
        labels = self._build_labels(encoded, masked_data['labels'])
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': labels
        }

    def _get_static_labels(self, sample: Dict) -> List[Dict]:
        """从原始标注生成静态标签"""
        labels = []
        text = sample['original_text'].split()
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
        return labels

    def _build_labels(self, encoded: Dict, mask_labels: List) -> torch.Tensor:
        """构建动态标签张量"""
        labels = torch.full_like(encoded['input_ids'], -100)
        tokenized_text = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        for label in mask_labels:
            pos = label['position']
            if pos < len(tokenized_text):
                # 处理subword情况
                if tokenized_text[pos].startswith('##'):
                    labels[0][pos] = -100
                else:
                    labels[0][pos] = self.tokenizer.convert_tokens_to_ids(label['word'])
        return labels.squeeze()

def get_dynamic_loader(file_paths: List[str], 
                      batch_size: int = 32,
                      strategy: str = "progressive") -> DataLoader:
    """获取动态掩码数据加载器"""
    dataset = DynamicMotionDataset(file_paths, mask_strategy=strategy)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }
    )

# 使用示例
if __name__ == "__main__":
    # 初始化加载器（支持多种策略）
    dir = "dataset/HumanML3D/texts_mask_deepseek"
    file_path_list = os.listdir(dir)
    file_paths = [os.path.join(dir, file_path) for file_path in file_path_list if file_path.endswith('.json')]
    loader = get_dynamic_loader(
        file_paths=file_paths,
        strategy="advanced"  # 可选: "random", "basic", "medium", "advanced"
    )
    
    # 获取一个批次
    batch = next(iter(loader))
    
    # 查看数据
    print("动态批次形状:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Labels: {batch['labels'].shape}")
    
    # 解码示例
    sample_idx = 0
    print("\n解码样例:")
    print("原始文本:", loader.dataset.raw_samples[sample_idx]['original_text'])
    # print("掩码文本:", tokenizer.decode(batch['input_ids'][sample_idx]))
    print("有效标签位置:", torch.where(batch['labels'][sample_idx] != -100)[0])

# if __name__ == "__main__":
#     # 使用示例
#     masker = ProgressiveMasker()
    
#     # 配置处理参数
#     custom_config = {
#         'stages': {
#             'basic': {'core': 0.4, 'dir': 0.0, 'body': 0.0},
#             'advanced': {'core': 0.8, 'dir': 0.6, 'body': 0.4}
#         },
#         'max_samples_per_file': 5000
#     }
    
#     # 运行处理流程
#     masker.process_dataset(
#         input_dir='dataset/HumanML3D/texts_mask_deepseek',
#         output_dir='dataset/HumanML3D/texts_mask_data'
#     )