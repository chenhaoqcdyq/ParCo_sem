import argparse, os, sys, glob
import torch
import time
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import repeat
import importlib
sys.path.append("/workspace/motion_diffusion/V19/FineTuneOpenMV_v17_DAC_basedonofficial")
from src.Open_MAGVIT2.modules.transformer.dac_gpt import *
# from src.IBQ.modules.transformer.dac_gpt import sample_IBQ
import time
from sample import *
import pdb
from transformers import AutoTokenizer
# sys.path.append("/mnt/workspace/wjc/Projects/X2A/ReproducedModels/descript-audio-codec")
import json
from main import *

class Ours_MAGVIT2:
    def __init__(self, config_path = "/workspace/motion_diffusion/V19/FineTuneOpenMV_v17_DAC_basedonofficial/configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_B.yaml", ckpt_path = "/workspace/motion_diffusion/V19/0417/0/epoch=292-step=95811.ckpt"):
        self.config = OmegaConf.load(config_path)
        self.ckpt = ckpt_path
        self.model, self.global_step = load_model(self.config, self.ckpt, gpu=True, eval_mode=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def sample(self, prompt, steps=197, temperature=[1, 1, 1, 1, 1, 1], top_k=[1, 1, 1, 1, 1, 1], top_p=[1, 1, 1, 1, 1, 1], cfg_scale=[1.75, 1.75, 1.75, 1.75, 1.75, 1.75]):
        textual_tokens = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        effect_text_length = textual_tokens['attention_mask'].sum().item()
        textual_tokens = textual_tokens['input_ids']
        if len(textual_tokens.shape) == 1:
            textual_tokens = textual_tokens.unsqueeze(0)
        textual_tokens = textual_tokens.to(self.model.device)
        # null = torch.zeros_like(textual_tokens)
        # textual_tokens = torch.cat([textual_tokens, null], dim=0)
        
        samples = dac_sample_Open_MAGVIT2(
                textual_tokens, self.model.transformer,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale)
        samples = [torch.cat(samples[i], dim=1) for i in range(6)]
        sequence = torch.stack(samples, dim=1).detach().cpu().squeeze()
        return sequence
if __name__ == "__main__":
    config = OmegaConf.load("/workspace/motion_diffusion/V19/FineTuneOpenMV_v17_DAC_basedonofficial/configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_B.yaml") #since only one config
    # ckpt = "/data/fengyuxiang/workspace/checkpoints/V17/1/epoch=299-step=98100.ckpt"
    ckpt = "/workspace/motion_diffusion/V19/0417/0/epoch=292-step=95811.ckpt"
    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True)



    dm = AudioCapsDataModule(
        batch_size=1, data_root="/dataset/motion/HumanML3D/vqvae_code11_lg0_train",
        max_length_m=197, max_length_t=64
    )
    dm.setup()
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()

    tra_root = "/dataset/motion/HumanML3D/gen/vqvae_code11_lg0_train"
    os.makedirs(tra_root, exist_ok=True)
    os.makedirs(tra_root+'/Train', exist_ok=True)
    os.makedirs(tra_root+'/Val', exist_ok=True)

    for num, loader in enumerate([train_loader]):
        mode = 'Train'#'Train' if num == 0 else 'Val'
        for sample in tqdm(loader):
            file_name = sample['name'][0]
            c = sample['prompt']
            gt = sample['dac']
            eff_t_length = sample['text_length'][0]
            # c = c[:, :eff_t_length]

            textual_tokens = c.to(model.device)
            null = torch.zeros_like(textual_tokens)
            textual_tokens = torch.cat([textual_tokens, null], dim=0)
            print("textual_tokens.shape = ",textual_tokens.shape)
            samples = dac_sample_Open_MAGVIT2(
                    textual_tokens, model.transformer,
                    steps=197,
                    temperature=[1, 1, 1, 1, 1, 1], 
                    top_k=[1, 1, 1, 1, 1, 1], 
                    top_p=[1, 1, 1, 1, 1, 1],
                    cfg_scale=[1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
            )
            # print("samples", samples)
            # sequence = torch.stack(samples, dim=1)[0].detach().cpu().squeeze()
            samples = [torch.cat(samples[i], dim=1) for i in range(6)]
            sequence = torch.stack(samples, dim=1).detach().cpu().squeeze()
            # print(sequence.shape)
            acc = (sequence[0] == gt.squeeze()).float().mean()
            # print(sequence[1], gt.squeeze())
            print(acc*100)
            np.save(f'{tra_root}/{mode}/{file_name}.npy', sequence)