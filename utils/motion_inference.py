import os
import torch
import numpy as np
from tqdm import tqdm
import sys
import json
from omegaconf import OmegaConf
import clip

# 添加必要的路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.motion_process import recover_from_ric
import visualize.plot_3d_global as plot_3d

class MotionInference:
    """
    动作推理类，用于从文本生成动作序列
    """
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        初始化动作推理类
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 检查点文件路径
            device: 计算设备，默认为'cuda'
        """
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        self.model, self.global_step = self._load_model()
        
        # 加载CLIP模型
        self.clip_model, _ = clip.load("ViT-B/32", device=torch.device(device), jit=False)
        clip.model.convert_weights(self.clip_model)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
            
        # 加载均值和标准差
        self.mean = None
        self.std = None
        
    def _load_model(self):
        """
        加载模型
        
        Returns:
            model: 加载的模型
            global_step: 全局步数
        """
        # 这里需要根据实际情况实现模型加载逻辑
        # 示例代码，需要根据实际项目调整
        from main import load_model
        model, global_step = load_model(self.config, self.checkpoint_path, gpu=True, eval_mode=True)
        return model, global_step
    
    def load_mean_std(self, mean_path, std_path):
        """
        加载均值和标准差
        
        Args:
            mean_path: 均值文件路径
            std_path: 标准差文件路径
        """
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        
    def infer_from_text(self, text, temperature=None, top_k=None, top_p=None, cfg_scale=None, 
                        steps=197, save_path=None, visualize=False):
        """
        从文本生成动作序列
        
        Args:
            text: 输入文本
            temperature: 温度参数列表，默认为[1, 1, 1, 1, 1, 1]
            top_k: top-k参数列表，默认为[1, 1, 1, 1, 1, 1]
            top_p: top-p参数列表，默认为[1, 1, 1, 1, 1, 1]
            cfg_scale: cfg-scale参数列表，默认为[1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
            steps: 生成步数，默认为197
            save_path: 保存路径，默认为None
            visualize: 是否可视化，默认为False
            
        Returns:
            sequence: 生成的动作序列
        """
        # 设置默认参数
        if temperature is None:
            temperature = [1, 1, 1, 1, 1, 1]
        if top_k is None:
            top_k = [1, 1, 1, 1, 1, 1]
        if top_p is None:
            top_p = [1, 1, 1, 1, 1, 1]
        if cfg_scale is None:
            cfg_scale = [1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
            
        # 处理文本
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        
        # 生成动作序列
        # 这里需要根据实际情况实现生成逻辑
        # 示例代码，需要根据实际项目调整
        from src.Open_MAGVIT2.modules.transformer.dac_gpt import dac_sample_Open_MAGVIT2
        
        # 准备输入
        textual_tokens = text_features
        null = torch.zeros_like(textual_tokens)
        textual_tokens = torch.cat([textual_tokens, null], dim=0)
        
        # 生成动作序列
        samples = dac_sample_Open_MAGVIT2(
            textual_tokens, self.model.transformer,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale
        )
        
        # 处理生成结果
        samples = [torch.cat(samples[i], dim=1) for i in range(6)]
        sequence = torch.stack(samples, dim=1).detach().cpu().squeeze()
        
        # 保存结果
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, sequence)
            
            # 可视化
            if visualize and self.mean is not None and self.std is not None:
                # 恢复3D坐标
                joints_num = 24
                positions = recover_from_ric((sequence * self.std + self.mean).float(), joints_num)
                
                # 保存为npy文件
                npy_path = save_path.replace('.npy', '_xyz.npy')
                np.save(npy_path, positions)
                
                # 可视化
                gif_path = save_path.replace('.npy', '.gif')
                plot_3d.draw_to_batch([positions], [text], [gif_path])
                
        return sequence
    
    def batch_inference(self, data_loader, output_dir, mode='Train', visualize=False):
        """
        批量推理
        
        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
            mode: 模式，默认为'Train'
            visualize: 是否可视化，默认为False
            
        Returns:
            results: 推理结果列表
        """
        results = []
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
        
        for sample in tqdm(data_loader, desc=f"Inferring {mode} samples"):
            file_name = sample['name'][0]
            text = sample['prompt']
            gt = sample['dac']
            eff_t_length = sample['text_length'][0]
            
            # 生成动作序列
            sequence = self.infer_from_text(
                text, 
                save_path=os.path.join(output_dir, mode, f"{file_name}.npy"),
                visualize=visualize
            )
            
            # 计算准确率
            if gt is not None:
                acc = (sequence[0] == gt.squeeze()).float().mean()
                print(f"Accuracy: {acc*100:.2f}%")
            
            results.append({
                'file_name': file_name,
                'sequence': sequence,
                'text': text
            })
            
        return results
    
    def evaluate_motion(self, pred_motion_list, eval_wrapper, top_k=3, text_list=None, save_dir=None, visualize=False):
        """
        评估生成的动作
        
        Args:
            pred_motion_list: 预测动作列表
            eval_wrapper: 评估包装器
            top_k: 计算top-k准确率时的k值
            text_list: 文本列表
            save_dir: 保存目录
            visualize: 是否可视化
            
        Returns:
            metrics: 评估指标
        """
        from utils.eval_bodypart import calculate_motion_metrics
        
        # 计算评估指标
        metrics = calculate_motion_metrics(
            pred_motion_list, 
            eval_wrapper, 
            top_k=top_k, 
            text_list=text_list,
            save_path=save_dir if visualize else None
        )
        
        return metrics 