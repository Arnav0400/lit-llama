import torch
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
from evolution_utils import EvolutionSearcher

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='mmlu')
    parser.add_argument('--save_path', type=str, default='out/evolution/mmlu10%/')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=2.00)
    parser.add_argument('--min-param-limits', type=float, default=0)
    args = parser.parse_args()
    seed = args.seed
    set_seed(seed)
    device = torch.device('cuda:0')
    args.best_acc = 0
    args.save_path = args.save_path
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    choices = dict()
    choices['A'] = ['LoRA_4', 'vector', 'constant', 'none']
    choices['B'] = ['LoRA_4', 'vector', 'constant', 'none']
    choices['C'] = ['LoRA_4', 'vector', 'none']
    choices['D'] = ['constant', 'none', 'vector']
    choices['E'] = ['constant', 'none', 'vector']
    searcher = EvolutionSearcher(args, choices, args.save_path)
    searcher.search()