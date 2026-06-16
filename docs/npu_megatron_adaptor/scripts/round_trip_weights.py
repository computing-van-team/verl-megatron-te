"""mbridge round-trip: HF -> megatron(megatron_adaptor/TE) -> HF, compare vs original HF.

Pinpoints which tensor the megatron<->HF weight conversion (mbridge, vanilla path)
gets wrong — the confirmed root cause of GRPO rollout_probs_diff 0.55 / corr 0.21.

Run: torchrun --standalone --nnodes=1 --nproc-per-node=1 round_trip_weights.py
"""
import os
import torch
import megatron_adaptor  # noqa: F401  (NPU monkey patches; must precede megatron import)
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

MODEL = "/home/b84412626/models/Qwen3-0.6B"

# ---- distributed / megatron init (single card, TP1 PP1) ----
dist.init_process_group(backend="hccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.npu.set_device(local_rank)
mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
model_parallel_cuda_manual_seed(42)

# ---- build megatron model + load HF weights via mbridge (vanilla path, as verl does) ----
from transformers import AutoConfig
from mbridge import AutoBridge

hf_config = AutoConfig.from_pretrained(MODEL)
bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16)
tf = bridge.config
tf.bf16 = True
models = bridge.get_model(bf16=True, wrap_with_ddp=False)
bridge.load_weights(models, MODEL)              # HF -> megatron
print(f"[ok] model built + weights loaded, chunks={len(models)}")

# ---- export back to HF format ----
exported = {}
for name, tensor in bridge.export_weights(models):  # vanilla path: yields HF names via _weight_to_hf_format
    exported[name] = tensor.detach().to(torch.float32).cpu()
print(f"[ok] exported {len(exported)} HF tensors")

# ---- load original HF state dict ----
from safetensors.torch import load_file
import glob
hf = {}
for f in glob.glob(MODEL + "/*.safetensors"):
    hf.update(load_file(f))
print(f"[ok] original HF tensors: {len(hf)}")

# ---- compare ----
only_export = sorted(set(exported) - set(hf))
only_hf = sorted(set(hf) - set(exported))
common = sorted(set(exported) & set(hf))
print(f"\n=== key coverage ===\nexport-only: {len(only_export)}  hf-only: {len(only_hf)}  common: {len(common)}")
for k in only_export[:8]:
    print("  EXPORT-ONLY:", k)
for k in only_hf[:8]:
    print("  HF-ONLY    :", k)

diffs = []
for k in common:
    a, b = exported[k], hf[k].float()
    if a.shape != b.shape:
        diffs.append((float("inf"), k, f"shape exp{tuple(a.shape)} hf{tuple(b.shape)}"))
        continue
    d = (a - b).abs().max().item()
    rel = d / (b.abs().max().item() + 1e-9)
    diffs.append((d, k, f"max|d|={d:.5f} rel={rel:.4f}"))

diffs.sort(key=lambda x: -x[0])
print("\n=== top-20 tensor diffs (export vs HF) ===")
for d, k, msg in diffs[:20]:
    print(f"  {msg:42s} {k}")
n_bad = sum(1 for d, _, _ in diffs if d > 1e-2)
print(f"\n=== {n_bad}/{len(common)} tensors with max|d|>1e-2 (bf16 noise is ~1e-2..1e-3) ===")
