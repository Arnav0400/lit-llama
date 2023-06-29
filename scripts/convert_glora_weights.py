import sys
import time
import copy
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn as nn

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.glora import glora

def merge_glora_state_dict(glora_model: nn.Module, merged_model: nn.Module):
    base_model_dict = glora_model.state_dict()
    layers = []
    for (name,l1),( _,l2) in zip(glora_model.named_modules(), merged_model.named_modules()):
        if 'c_attn' in name:
            path_config = l1.eval_config
            A = l1.prepare_path(path_config['A'], l1.Ad, l1.Au).cpu()
            B = l1.prepare_path(path_config['B'], l1.Bd, l1.Bu).cpu()  
            C = l1.prepare_path(path_config['C'], l1.Cd, l1.Cu).cpu()
            D = l1.prepare_path(path_config['D'], l1.D).cpu()
            E = l1.prepare_path(path_config['E'], l1.E).cpu()
            l2.weight.data = l1.weight.data + l1.weight.data*A + B
            l2.bias = nn.Parameter(E+torch.matmul(l1.weight, C).squeeze())
    return merged_model

def main(
    accelerator: str = "cpu",
    glora_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    dtype: str = "float32",
) -> None:
    """Merges glora weights to base model.

    Args:
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        glora_path: Path to the checkpoint with trained GLoRA weights, which are the output of
            `finetune_glora.py`.
        checkpoint_path: The checkpoint path to load.
        config_path: Path to the evolutionary search output ckpt
        dtype: `torch.dtype` to work with
    """
    if not glora_path:
        glora_path = Path("/nfs/users/ext_arnav.chavan/NIPS23/lit-llama/out/glora/alpaca/iter-179199-ckpt.pth")
    if not checkpoint_path:
        checkpoint_path = Path(f"/nfs/users/ext_arnav.chavan/NIPS23/lit-llama/checkpoints/lit-llama/7B/lit-llama.pth")
    if not config_path:
        config_path = Path(f"/nfs/users/ext_arnav.chavan/NIPS23/lit-llama/out/evolution/mmlu/checkpoint-5.pth.tar")

    assert glora_path.is_file()
    assert checkpoint_path.is_file()
    assert config_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with (lazy_load(checkpoint_path) as pretrained_checkpoint,
          lazy_load(glora_path) as glora_checkpoint,
          lazy_load(config_path) as config_checkpoint):
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
            device=fabric.device, dtype=dtype
            ):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            merged_model = copy.deepcopy(model)
            # 2. Load the fine-tuned glora weights
            if glora_path:
                glora(model, 4)
                glora_checkpoint = torch.load(glora_path)
                glora_checkpoint_mod = {}
                for n in glora_checkpoint:
                    glora_checkpoint_mod['.'.join(n.split('.')[1:])] = glora_checkpoint[n]
                model.load_state_dict(glora_checkpoint_mod, strict=False)

            if config_path:
                i = 0
                ckpt = torch.load(config_path)
                config = ckpt['keep_top_k'][50][0]
                for name, l in model.named_modules():
                    if 'c_attn' in name:
                        l.eval_config = config[i]
                        i+=1
                print(f'Setup config for {i} layers')

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    merged_model = merge_glora_state_dict(model, merged_model)
    save_path = glora_path.with_stem(f"{glora_path.stem}-glora-merged-weights-10%-{str(config_path.stem).split('/')[-1].split('.')[0]}")
    print("Saving GLoRA to base model weights ...")
    torch.save(merged_model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)