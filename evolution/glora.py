import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, List
import random

import lit_llama.model as llama

from contextlib import contextmanager
from dataclasses import dataclass


class GLoRALayer():
    def __init__(
        self, 
        r: int, 
    ):
        self.r = r
        config_A_B = [f'LoRA_{self.r}', 'vector', 'constant', 'none']
        config_C = [f'LoRA_{self.r}', 'vector', 'none']
        config_D_E = ['constant', 'none', 'vector']
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)

class MergedLinear(nn.Linear, GLoRALayer):
    # GLoRA implemented in a dense layer
    def __init__(
        self, 
        # ↓ this part is for pretrained weights
        in_features: int, 
        out_features: int, 
        # ↓ the remaining part is for LoRA
        r: int = 0, 
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GLoRALayer.__init__(self, r=r)
        self.Ad, self.Au = self.make_param((out_features, in_features), f'LoRA_{self.r}')
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{self.r}')
        self.Cd, self.Cu = self.make_param((in_features, 1), f'LoRA_{self.r}')
        self.D = nn.Parameter(torch.zeros(out_features))
        self.E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.kaiming_uniform_(self.Au, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Bu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Cu, a=math.sqrt(5))
    
    def make_param(self, shape, config=None):
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))
    
    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            elif 'none' in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1]).cuda()
            else:
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1).cuda()
            else:
                raise ValueError
        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config['A'], self.Ad, self.Au)
        B = self.prepare_path(path_config['B'], self.Bd, self.Bu)
        C = self.prepare_path(path_config['C'], self.Cd, self.Cu)
        D = self.prepare_path(path_config['D'], self.D)
        E = self.prepare_path(path_config['E'], self.E)
        optimal_weight = self.weight + self.weight*A + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias + self.bias*D + E
        else:
            optimal_bias = E
        optimal_prompt = torch.matmul(self.weight, C).squeeze()
        return F.linear(x, optimal_weight, optimal_bias+optimal_prompt)

    @staticmethod
    def from_linear(linear_module, r):
        new_linear = MergedLinear(linear_module.in_features, linear_module.out_features, r=r)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class ModuleInjection:

    @staticmethod
    def make_scalable(linear_module, r):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a super linear that can be trained to
        """
        new_linear = MergedLinear.from_linear(linear_module, r)
        return new_linear
    
def mark_only_glora_as_trainable(model: nn.Module) -> None:
    total_trainable_param = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            total_trainable_param += p.numel()
            p.requires_grad = True
        else:
            p.requires_grad = False
    return total_trainable_param

def glora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            trainable[n] = p
    return trainable

def glora(model, r):
    layers = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if 'attn' in last_token:
            weightb4 = getattr(parent_layer, last_token).weight.sum()
            setattr(parent_layer, last_token, ModuleInjection.make_scalable(getattr(parent_layer, last_token),r))
            weightafter = getattr(parent_layer, last_token).weight.sum()
            assert weightb4==weightafter
